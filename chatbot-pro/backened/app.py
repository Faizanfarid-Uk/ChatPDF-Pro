from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import traceback

# Import RAG chain functions
from rag_chain import (
    load_pdf_from_user,
    split_documents,
    create_vector_store,
    build_conversational_rag_chain,
    store
)

# =======================
# APP CONFIGURATION
# =======================

FRONTEND_FOLDER = os.path.abspath("../frontend")

app = Flask(
    __name__,
    static_folder=os.path.join(FRONTEND_FOLDER, "static"),
    template_folder=FRONTEND_FOLDER
)

CORS(app)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
ALLOWED_EXTENSIONS = {"pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

# =======================
# GLOBAL STATE
# =======================

rag_chain = None
vector_store = None

# =======================
# HELPERS
# =======================

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# =======================
# ROUTES
# =======================

@app.route("/", strict_slashes=False)
def home():
    try:
        return render_template("index.html")
    except Exception:
        return jsonify({
            "message": "Backend is running",
            "endpoints": [
                "POST /upload",
                "POST /chat",
                "POST /login",
                "GET /health",
                "GET /stats"
            ]
        })


@app.route("/login", methods=["GET", "POST"], strict_slashes=False)
def login():
    if request.method == "GET":
        try:
            return render_template("login.html")
        except Exception:
            return jsonify({"error": "login.html not found"}), 404

    data = request.json or {}
    email = data.get("email")
    password = data.get("password")

    if email and password:
        return jsonify({
            "success": True,
            "message": "Login successful",
            "user": {"email": email}
        })

    return jsonify({"success": False, "error": "Invalid credentials"}), 401


@app.route("/upload", methods=["GET", "POST"], strict_slashes=False)
def upload_pdf():
    global rag_chain, vector_store

    if request.method == "GET":
        try:
            return render_template("upload.html")
        except Exception:
            return jsonify({"message": "Upload endpoint ready"})

    if "pdf" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    pdf_file = request.files["pdf"]

    if pdf_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(pdf_file.filename):
        return jsonify({"error": "Only PDF files allowed"}), 400

    filename = secure_filename(pdf_file.filename)
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        pdf_file.save(pdf_path)

        docs = load_pdf_from_user(pdf_path)
        splits = split_documents(docs)

        vector_store = create_vector_store(splits)
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        rag_chain = build_conversational_rag_chain(retriever)

        return jsonify({
            "success": True,
            "message": "PDF processed successfully",
            "filename": filename,
            "pages": len(docs),
            "chunks": len(splits)
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"], strict_slashes=False)
def chat():
    global rag_chain

    if rag_chain is None:
        return jsonify({
            "success": False,
            "error": "Upload a PDF first"
        }), 400

    data = request.json or {}
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    try:
        response = rag_chain.invoke(
            {"input": user_message},
            config={"configurable": {"session_id": session_id}}
        )

        answer = response.get("answer", "No answer generated")

        return jsonify({
            "success": True,
            "answer": answer,
            "session_id": session_id
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/reset", methods=["POST"], strict_slashes=False)
def reset():
    data = request.json or {}
    session_id = data.get("session_id", "default")

    if session_id in store:
        store[session_id].clear()

    return jsonify({
        "success": True,
        "message": "Conversation reset",
        "session_id": session_id
    })


@app.route("/stats", methods=["GET"], strict_slashes=False)
def stats():
    vector_size = 0
    if vector_store:
        try:
            vector_size = len(vector_store.index_to_docstore_id)
        except Exception:
            pass

    return jsonify({
        "documents_processed": 1 if rag_chain else 0,
        "vector_store_size": vector_size,
        "active_sessions": len(store),
        "system_status": "online"
    })


@app.route("/health", methods=["GET"], strict_slashes=False)
def health():
    return jsonify({
        "status": "healthy",
        "rag_initialized": rag_chain is not None,
        "vector_store_ready": vector_store is not None
    })


# =======================
# RUN SERVER
# =======================

if __name__ == "__main__":
    print("=" * 60)
    print("NexusAI Backend Server Running")
    print("URL: http://127.0.0.1:5000")
    print("=" * 60)

    app.run(debug=True, host="0.0.0.0", port=5000)
