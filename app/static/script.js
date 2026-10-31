/**
 * PDF Chatbot - Enhanced Frontend Script
 * Handles drag-and-drop, file upload, question asking, suggestion chips, and clipboard copying.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements - Upload Section
    const fileInput = document.getElementById("pdf-file-input");
    const dropzone = document.getElementById("dropzone");
    const fileSelectedInfo = document.getElementById("file-selected-info");
    const selectedFilename = document.getElementById("selected-filename");
    const clearFileBtn = document.getElementById("clear-file-btn");
    const uploadForm = document.getElementById("upload-form");
    const uploadBtn = document.getElementById("upload-btn");
    const uploadSpinner = uploadBtn.querySelector(".btn-spinner");
    const uploadBtnText = uploadBtn.querySelector(".btn-text");
    const uploadAlert = document.getElementById("upload-alert");
    const uploadAlertIcon = document.getElementById("upload-alert-icon");
    const uploadAlertMessage = document.getElementById("upload-alert-message");
    const docMetaBadge = document.getElementById("doc-meta-badge");
    const metaDocName = document.getElementById("meta-doc-name");
    const metaChunksCount = document.getElementById("meta-chunks-count");

    // DOM Elements - Question Section
    const askForm = document.getElementById("ask-form");
    const questionInput = document.getElementById("question-input");
    const askBtn = document.getElementById("ask-btn");
    const askSpinner = askBtn.querySelector(".btn-spinner");
    const askBtnText = askBtn.querySelector(".btn-text");
    const askHint = document.getElementById("ask-hint");
    const loadingContainer = document.getElementById("loading-container");
    const qaAlert = document.getElementById("qa-alert");
    const qaAlertMessage = document.getElementById("qa-alert-message");
    const answerSection = document.getElementById("answer-section");
    const answerContent = document.getElementById("answer-content");
    const sourcesList = document.getElementById("sources-list");
    const copyBtn = document.getElementById("copy-answer-btn");
    const suggestionChips = document.querySelectorAll(".suggestion-chip");

    let currentFile = null;

    // Check health status on load to restore state if document already indexed
    checkBackendHealth();

    async function checkBackendHealth() {
        try {
            const res = await fetch("/health");
            if (!res.ok) return;
            const data = await res.json();
            if (data.indexed_document) {
                enableQuestionInput(data.indexed_document, data.total_chunks);
            }
        } catch (e) {
            console.log("Healthcheck status:", e);
        }
    }

    // =========================================================================
    // File Selection & Drag-and-Drop
    // =========================================================================

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    ["dragenter", "dragover"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    function handleFileSelection(file) {
        if (!file.name.toLowerCase().endsWith(".pdf")) {
            showUploadAlert("error", "Only PDF (.pdf) files are supported.");
            return;
        }

        currentFile = file;
        selectedFilename.textContent = file.name;
        selectedFilename.setAttribute("title", file.name);
        fileSelectedInfo.classList.remove("hidden");
        dropzone.classList.add("hidden");
        uploadBtn.disabled = false;
        hideUploadAlert();
    }

    clearFileBtn.addEventListener("click", () => {
        currentFile = null;
        fileInput.value = "";
        selectedFilename.textContent = "";
        fileSelectedInfo.classList.add("hidden");
        dropzone.classList.remove("hidden");
        uploadBtn.disabled = true;
        hideUploadAlert();
    });

    // =========================================================================
    // Upload PDF Submission
    // =========================================================================

    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!currentFile) return;

        setUploadLoading(true);
        hideUploadAlert();

        const formData = new FormData();
        formData.append("file", currentFile);

        try {
            const response = await fetch("/upload", {
                method: "POST",
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || "Failed to process the PDF document.");
            }

            // Show success message
            showUploadAlert("success", result.message || "PDF uploaded and processed successfully!");
            enableQuestionInput(result.filename, result.total_chunks);

        } catch (error) {
            showUploadAlert("error", error.message);
        } finally {
            setUploadLoading(false);
        }
    });

    function setUploadLoading(isLoading) {
        uploadBtn.disabled = isLoading;
        if (isLoading) {
            uploadSpinner.classList.remove("hidden");
            uploadBtnText.textContent = "Processing PDF...";
        } else {
            uploadSpinner.classList.add("hidden");
            uploadBtnText.textContent = "Upload & Process PDF";
        }
    }

    function showUploadAlert(type, message) {
        uploadAlert.className = `alert alert-${type}`;
        uploadAlertIcon.textContent = type === "success" ? "✅" : "⚠️";
        uploadAlertMessage.textContent = message;
        uploadAlert.classList.remove("hidden");
    }

    function hideUploadAlert() {
        uploadAlert.classList.add("hidden");
    }

    function enableQuestionInput(filename, chunksCount) {
        // Update document metadata badge with zero overflow guarantee
        metaDocName.textContent = filename;
        metaDocName.setAttribute("title", filename);
        metaChunksCount.textContent = chunksCount;
        docMetaBadge.classList.remove("hidden");

        // Unlock question form
        questionInput.disabled = false;
        askBtn.disabled = false;
        askHint.textContent = `Active: ${filename}`;
        questionInput.focus();
    }

    // =========================================================================
    // Suggestion Chips
    // =========================================================================

    suggestionChips.forEach((chip) => {
        chip.addEventListener("click", () => {
            if (questionInput.disabled) {
                showQaAlert("Please upload and index a PDF before selecting suggested questions.");
                return;
            }
            const promptText = chip.getAttribute("data-prompt");
            questionInput.value = promptText;
            askForm.dispatchEvent(new Event("submit"));
        });
    });

    // =========================================================================
    // Ask Question Submission
    // =========================================================================

    askForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const question = questionInput.value.trim();
        if (!question) return;

        setAskLoading(true);
        hideQaAlert();
        answerSection.classList.add("hidden");

        try {
            const response = await fetch("/ask", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ question })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "Failed to generate answer.");
            }

            // Display Answer
            answerContent.textContent = data.answer;

            // Display Source Chunks
            renderSources(data.sources || []);

            answerSection.classList.remove("hidden");

        } catch (error) {
            showQaAlert(error.message);
        } finally {
            setAskLoading(false);
        }
    });

    function renderSources(sources) {
        sourcesList.innerHTML = "";

        if (sources.length === 0) {
            const emptyNotice = document.createElement("p");
            emptyNotice.className = "source-text";
            emptyNotice.textContent = "No specific source passages were cited for this response.";
            sourcesList.appendChild(emptyNotice);
            return;
        }

        sources.forEach((source) => {
            const card = document.createElement("div");
            card.className = "source-card";

            const header = document.createElement("div");
            header.className = "source-header";
            header.innerHTML = `
                <span class="source-rank">Chunk #${source.rank}</span>
                <span class="source-score">L2 Dist: ${source.score}</span>
            `;

            const text = document.createElement("div");
            text.className = "source-text";
            text.textContent = `"${source.text}"`;

            card.appendChild(header);
            card.appendChild(text);
            sourcesList.appendChild(card);
        });
    }

    // =========================================================================
    // Copy Answer to Clipboard
    // =========================================================================

    if (copyBtn) {
        copyBtn.addEventListener("click", async () => {
            const textToCopy = answerContent.textContent;
            if (!textToCopy) return;

            try {
                await navigator.clipboard.writeText(textToCopy);
                const copyText = copyBtn.querySelector(".copy-text");
                const originalText = copyText.textContent;
                copyText.textContent = "Copied!";
                copyBtn.style.borderColor = "var(--success-green)";
                copyBtn.style.color = "var(--success-green)";

                setTimeout(() => {
                    copyText.textContent = originalText;
                    copyBtn.style.borderColor = "";
                    copyBtn.style.color = "";
                }, 2000);
            } catch (err) {
                console.error("Clipboard copy failed:", err);
            }
        });
    }

    function setAskLoading(isLoading) {
        askBtn.disabled = isLoading;
        questionInput.disabled = isLoading;
        if (isLoading) {
            askSpinner.classList.remove("hidden");
            askBtn.querySelector("span:not(.btn-spinner)").classList.add("hidden");
            loadingContainer.classList.remove("hidden");
        } else {
            askSpinner.classList.add("hidden");
            askBtn.querySelector("span:not(.btn-spinner)").classList.remove("hidden");
            loadingContainer.classList.add("hidden");
            questionInput.disabled = false;
        }
    }

    function showQaAlert(message) {
        qaAlertMessage.textContent = message;
        qaAlert.classList.remove("hidden");
    }

    function hideQaAlert() {
        qaAlert.classList.add("hidden");
    }
});
