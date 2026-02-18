document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const jdText = document.getElementById('jdText');
    const resumeFile = document.getElementById('resumeFile');
    const dropZone = document.getElementById('dropZone');
    const filePreview = document.getElementById('filePreview');
    const fileName = document.getElementById('fileName');
    const removeFile = document.getElementById('removeFile');
    const submitBtn = document.getElementById('submitBtn');
    const progressCard = document.getElementById('progressCard');
    const downloadCard = document.getElementById('downloadCard');
    const regenerateBtn = document.getElementById('regenerateBtn');
    const fileError = document.getElementById('file-error');
    const jdError = document.getElementById('jd-error');

    // ── Drag & Drop ─────────────────────────────────────────────────
    dropZone.addEventListener('click', (e) => {
        if (e.target.closest('.remove-file')) return;
        resumeFile.click();
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length && files[0].type === 'application/pdf') {
            resumeFile.files = e.dataTransfer.files;
            showFilePreview(files[0]);
        } else {
            showError(fileError, 'Please drop a PDF file.');
        }
    });

    resumeFile.addEventListener('change', () => {
        if (resumeFile.files.length) {
            showFilePreview(resumeFile.files[0]);
            clearError(fileError);
        }
    });

    removeFile.addEventListener('click', (e) => {
        e.stopPropagation();
        resumeFile.value = '';
        filePreview.style.display = 'none';
        dropZone.querySelector('.drop-content').style.display = '';
    });

    function showFilePreview(file) {
        fileName.textContent = file.name;
        filePreview.style.display = 'flex';
        dropZone.querySelector('.drop-content').style.display = 'none';
    }

    // ── Form Submit ─────────────────────────────────────────────────
    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        // Validate
        let valid = true;
        if (!resumeFile.files.length) {
            showError(fileError, 'Please upload your resume PDF.');
            valid = false;
        }
        if (!jdText.value.trim()) {
            showError(jdError, 'Please enter a job description.');
            valid = false;
        }
        if (!valid) return;

        clearError(fileError);
        clearError(jdError);

        // Show progress, hide form
        submitBtn.disabled = true;
        progressCard.style.display = 'flex';
        downloadCard.style.display = 'none';

        const formData = new FormData();
        formData.append('jd_text', jdText.value.trim());
        formData.append('resume_pdf', resumeFile.files[0]);

        try {
            const response = await fetch('https://beula-doleritic-wispily.ngrok-free.dev/generate-resume', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                let detail = 'Server error';
                try {
                    const errData = await response.json();
                    detail = errData.detail || detail;
                } catch (_) { }
                throw new Error(detail);
            }

            // Download the PDF blob
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'tailored_resume.pdf';
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);

            // Show success
            progressCard.style.display = 'none';
            downloadCard.style.display = 'flex';

        } catch (err) {
            progressCard.style.display = 'none';
            showError(jdError, 'Something went wrong: ' + err.message);
        } finally {
            submitBtn.disabled = false;
        }
    });

    // ── Regenerate ──────────────────────────────────────────────────
    regenerateBtn.addEventListener('click', () => {
        downloadCard.style.display = 'none';
        jdText.value = '';
        resumeFile.value = '';
        filePreview.style.display = 'none';
        dropZone.querySelector('.drop-content').style.display = '';
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // ── Real-time validation ────────────────────────────────────────
    jdText.addEventListener('input', () => {
        if (jdText.value.trim()) clearError(jdError);
    });

    resumeFile.addEventListener('change', () => {
        if (resumeFile.files.length) clearError(fileError);
    });

    // ── Helpers ─────────────────────────────────────────────────────
    function showError(el, message) {
        el.textContent = message;
    }

    function clearError(el) {
        el.textContent = '';
    }
});
