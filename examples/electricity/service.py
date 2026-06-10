# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Flask service for hierarchical RAG retriever.

Endpoint:
    POST /retrieve
        Request JSON: {"question": "你的问题"}
        Response JSON: {"question": "...", "chunks": ["chunk1", "chunk2", ...], "count": 10}

    GET /retrieve?question=你的问题
        Response JSON: {"question": "...", "chunks": ["chunk1", "chunk2", ...], "count": 10}

    GET /health
        Response JSON: {"status": "ok"}
"""

import argparse
import os
import sys
import yaml
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from flask import Flask, jsonify, request

# Ensure project root is in sys.path so pikerag can be imported
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Query timeout in seconds (prevents hanging on LLM API calls)
QUERY_TIMEOUT = 120


def load_config(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as fin:
        try:
            return yaml.safe_load(fin, Loader=yaml.FullLoader)
        except TypeError:
            return yaml.safe_load(fin)


def _resolve_data_paths(retriever_args: dict, root_dir: str) -> dict:
    """Resolve relative paths in retriever config to absolute paths."""
    import copy
    args = copy.deepcopy(retriever_args)

    # Fix vector_store persist_directory
    vs = args.get("vector_store", {})
    if "persist_directory" in vs:
        persist_dir = vs["persist_directory"]
        if not os.path.isabs(persist_dir):
            vs["persist_directory"] = os.path.join(root_dir, persist_dir)

    # Fix id_document_loading filepath
    id_loading = vs.get("id_document_loading", {})
    if "args" in id_loading and "filepath" in id_loading["args"]:
        fp = id_loading["args"]["filepath"]
        if not os.path.isabs(fp):
            id_loading["args"]["filepath"] = os.path.join(root_dir, fp)

    return args


def build_retriever(config: dict, root_dir: str = None):
    from pikerag.utils.config_loader import load_dot_env, load_class
    from pikerag.utils.logger import Logger
    from pikerag.knowledge_retrievers import BaseQaRetriever

    if root_dir is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Load env
    load_dot_env(env_path=config.get("dotenv_path", None))

    # Setup log dir
    log_dir = os.path.join(root_dir, config["log_root_dir"], config["experiment_name"])
    os.makedirs(log_dir, exist_ok=True)

    logger = Logger(name=config["experiment_name"], dump_folder=os.path.join(root_dir, config["log_root_dir"]))

    # Build retriever config from YAML retriever section
    retriever_config = config["retriever"]
    retriever_args: dict = retriever_config.get("args", {})

    # Resolve relative data paths to absolute paths
    retriever_args = _resolve_data_paths(retriever_args, root_dir)

    # Merge answer_llm and entity_llm from top level into retriever args
    if "answer_llm" in config:
        answer_llm = config["answer_llm"]
        if "llm_config" in answer_llm:
            answer_llm_args = answer_llm.get("args", {})
            answer_llm_args["llm_config"] = answer_llm.get("llm_config", {})
            answer_llm = {**answer_llm, "args": answer_llm_args}
        retriever_args["answer_llm"] = answer_llm

    if "entity_llm" in config:
        entity_llm = config["entity_llm"]
        if "llm_config" in entity_llm:
            entity_llm_args = entity_llm.get("args", {})
            entity_llm_args["llm_config"] = entity_llm.get("llm_config", {})
            entity_llm = {**entity_llm, "args": entity_llm_args}
        retriever_args["entity_llm"] = entity_llm

    # Load retriever class
    retriever_class = load_class(
        module_path=retriever_config["module_path"],
        class_name=retriever_config["class_name"],
        base_class=BaseQaRetriever,
    )

    retriever = retriever_class(
        retriever_config=retriever_args,
        log_dir=log_dir,
        main_logger=logger,
    )

    print(f"Retriever '{retriever_class.name}' initialized successfully.")
    return retriever


# ==============================================================================
# Helpers
# ==============================================================================
def _parse_int(data: dict, key: str, default: int = None) -> int:
    if key in data and data[key] is not None and str(data[key]).strip() != "":
        return int(data[key])
    return default


def _parse_float(data: dict, key: str, default: float = None) -> float:
    if key in data and data[key] is not None and str(data[key]).strip() != "":
        return float(data[key])
    return default


def _parse_bool(data: dict, key: str, default: bool = None) -> bool:
    if key in data and data[key] is not None and str(data[key]).strip() != "":
        val = str(data[key]).strip().lower()
        return val in ("true", "1", "yes")
    return default


def _apply_overrides(retrieve_k, use_entity_ranking, entity_weight, similarity_weight) -> None:
    if retrieve_k is not None:
        _retriever.retrieve_k = retrieve_k
    if use_entity_ranking is not None:
        _retriever.use_entity_ranking = use_entity_ranking
    if entity_weight is not None:
        _retriever.entity_weight = entity_weight
    if similarity_weight is not None:
        _retriever.similarity_weight = similarity_weight


# ==============================================================================
# Globals - initialized once at startup
# ==============================================================================
_retriever = None
_config = None


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/retrieve", methods=["GET", "POST"])
    def retrieve():
        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            question = data.get("question", "")
        else:
            question = request.args.get("question", "")
            data = request.args

        if not question:
            return jsonify({"error": "question is required"}), 400

        # Parse optional overrides
        retrieve_k = _parse_int(data, "retrieve_k")
        use_entity_ranking = _parse_bool(data, "use_entity_ranking")
        entity_weight = _parse_float(data, "entity_weight")
        similarity_weight = _parse_float(data, "similarity_weight")

        # Apply overrides to retriever instance
        _apply_overrides(retrieve_k, use_entity_ranking, entity_weight, similarity_weight)

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    _retriever.retrieve_contents_by_query, question,
                    retrieve_k=_retriever.retrieve_k,
                )
                chunks = future.result(timeout=QUERY_TIMEOUT)
            return jsonify({
                "question": question,
                "chunks": chunks,
                "count": len(chunks),
                "method": "hierarchical",
                "params": {
                    "retrieve_k": _retriever.retrieve_k,
                    "use_entity_ranking": _retriever.use_entity_ranking,
                    "entity_weight": _retriever.entity_weight,
                    "similarity_weight": _retriever.similarity_weight,
                },
            })
        except (Exception, FutureTimeoutError) as e:
            # Fallback to basic vector search when LLM calls fail (e.g. no API key)
            try:
                chunk_infos = _retriever._get_doc_with_query(
                    question,
                    _retriever.vector_store,
                    _retriever.retrieve_k,
                    _retriever.retrieve_score_threshold,
                )
                chunks = _retriever._get_relevant_strings(chunk_infos, "")
                return jsonify({
                    "question": question,
                    "chunks": chunks,
                    "count": len(chunks),
                    "method": "basic_fallback",
                })
            except Exception as fallback_error:
                return jsonify({
                    "error": str(e),
                    "fallback_error": str(fallback_error),
                    "question": question,
                }), 500

    return app


# ==============================================================================
# Entry point
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval service for hierarchical RAG")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs", "hierarchical_rag_iter_hier_entity.yml"),
        help="Path to YAML config file",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    print(f"Loading config: {config_path}")
    _config = load_config(config_path)

    print(f"Project root: {_project_root}")
    print("Initializing retriever (this may take a moment)...")
    _retriever = build_retriever(_config, root_dir=_project_root)

    app = create_app()
    print(f"Service starting on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
