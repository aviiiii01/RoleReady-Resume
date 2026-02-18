document.addEventListener('DOMContentLoaded', () => {
    // ── Views ───────────────────────────────────────────────────────
    const viewLanding = document.getElementById('viewLanding');
    const viewUpdate = document.getElementById('viewUpdate');
    const viewScratch = document.getElementById('viewScratch');

    const btnModeUpdate = document.getElementById('btnModeUpdate');
    const btnModeScratch = document.getElementById('btnModeScratch');
    const backFromUpdate = document.getElementById('backFromUpdate');
    const backFromScratch = document.getElementById('backFromScratch');

    // ── Update form elements ────────────────────────────────────────
    const uploadForm = document.getElementById('uploadForm');
    const jdText = document.getElementById('jdText');
    const resumeFile = document.getElementById('resumeFile');
    const dropZone = document.getElementById('dropZone');
    const filePreview = document.getElementById('filePreview');
    const fileName = document.getElementById('fileName');
    const removeFile = document.getElementById('removeFile');
    const submitBtn = document.getElementById('submitBtn');
    const fileError = document.getElementById('file-error');
    const jdError = document.getElementById('jd-error');

    // ── Scratch form elements ───────────────────────────────────────
    const scratchForm = document.getElementById('scratchForm');
    const scratchSubmitBtn = document.getElementById('scratchSubmitBtn');
    const scratchJdError = document.getElementById('scratch-jd-error');

    // ── Shared elements ─────────────────────────────────────────────
    const progressCard = document.getElementById('progressCard');
    const atsProgressCard = document.getElementById('atsProgressCard');
    const downloadCard = document.getElementById('downloadCard');
    const regenerateBtn = document.getElementById('regenerateBtn');
    const atsSection = document.getElementById('atsSection');
    const atsRingFill = document.getElementById('atsRingFill');
    const atsScoreNumber = document.getElementById('atsScoreNumber');
    const atsSuggestions = document.getElementById('atsSuggestions');
    const suggestionsList = document.getElementById('suggestionsList');

    const API_BASE = 'https://beula-doleritic-wispily.ngrok-free.dev';

    // ── View Navigation ─────────────────────────────────────────────
    function showView(viewEl) {
        [viewLanding, viewUpdate, viewScratch].forEach(v => v.classList.add('hidden'));
        progressCard.style.display = 'none';
        atsProgressCard.style.display = 'none';
        downloadCard.style.display = 'none';
        viewEl.classList.remove('hidden');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    btnModeUpdate.addEventListener('click', () => showView(viewUpdate));
    btnModeScratch.addEventListener('click', () => showView(viewScratch));
    backFromUpdate.addEventListener('click', () => showView(viewLanding));
    backFromScratch.addEventListener('click', () => showView(viewLanding));

    // ── Dynamic "Add Entry" buttons ─────────────────────────────────
    document.querySelectorAll('.add-entry-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.dataset.target;
            const container = document.getElementById(targetId);
            const firstTextarea = container.querySelector('textarea');
            const entry = document.createElement('div');
            entry.classList.add('dynamic-entry');
            entry.innerHTML = `
                <textarea class="small-textarea" placeholder="${firstTextarea.placeholder}"></textarea>
                <button type="button" class="remove-entry-btn" title="Remove">&times;</button>
            `;
            container.appendChild(entry);

            entry.querySelector('.remove-entry-btn').addEventListener('click', () => {
                entry.remove();
            });
        });
    });

    // ── Drag & Drop (Update form) ───────────────────────────────────
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

    // ── Update Form Submit ──────────────────────────────────────────
    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault();

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

        submitBtn.disabled = true;
        viewUpdate.classList.add('hidden');
        progressCard.style.display = 'flex';

        const formData = new FormData();
        formData.append('jd_text', jdText.value.trim());
        formData.append('mode', 'update');
        formData.append('resume_pdf', resumeFile.files[0]);

        try {
            const blob = await generateResume(formData);
            autoDownload(blob);
            progressCard.style.display = 'none';
            atsProgressCard.style.display = 'flex';
            await fetchATSScore(blob, jdText.value.trim());
        } catch (err) {
            progressCard.style.display = 'none';
            viewUpdate.classList.remove('hidden');
            showError(jdError, 'Something went wrong: ' + err.message);
        } finally {
            submitBtn.disabled = false;
        }
    });

    // ── Scratch Form Submit ──────────────────────────────────────────
    scratchForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        const scratchJd = document.getElementById('scratchJd');
        if (!scratchJd.value.trim()) {
            showError(scratchJdError, 'Please enter a job description.');
            return;
        }
        clearError(scratchJdError);

        // Collect all scratch fields
        const name = document.getElementById('scratchName').value.trim();
        const email = document.getElementById('scratchEmail').value.trim();
        const phone = document.getElementById('scratchPhone').value.trim();
        const location = document.getElementById('scratchLocation').value.trim();
        const linkedin = document.getElementById('scratchLinkedin').value.trim();
        const github = document.getElementById('scratchGithub').value.trim();
        const portfolio = document.getElementById('scratchPortfolio').value.trim();
        const skills = document.getElementById('scratchSkills').value.trim();

        const education = collectEntries('educationEntries');
        const experience = collectEntries('experienceEntries');
        const projects = collectEntries('projectEntries');
        const achievements = collectEntries('achievementEntries');

        // Build a structured details string (same format the agent expects)
        const links = [linkedin, github, portfolio].filter(Boolean).join(', ');
        const candidateDetails = `
=== CANDIDATE DETAILS ===
Name: ${name}
Email: ${email}
Phone: ${phone}
Location: ${location}
Links: ${links || 'N/A'}

=== EDUCATION ===
${education.map(e => '- ' + e).join('\n') || 'N/A'}

=== EXPERIENCE ===
${experience.map(e => '- ' + e).join('\n') || 'N/A'}

=== PROJECTS ===
${projects.map(p => '- ' + p).join('\n') || 'N/A'}

=== SKILLS ===
${skills || 'N/A'}

=== ACHIEVEMENTS ===
${achievements.map(a => '- ' + a).join('\n') || 'N/A'}
        `.trim();

        // Send to backend
        scratchSubmitBtn.disabled = true;
        viewScratch.classList.add('hidden');
        progressCard.style.display = 'flex';

        const formData = new FormData();
        formData.append('jd_text', scratchJd.value.trim());
        formData.append('mode', 'scratch');
        formData.append('candidate_details', candidateDetails);

        try {
            const blob = await generateResume(formData);
            autoDownload(blob);
            progressCard.style.display = 'none';
            atsProgressCard.style.display = 'flex';
            await fetchATSScore(blob, scratchJd.value.trim());
        } catch (err) {
            progressCard.style.display = 'none';
            viewScratch.classList.remove('hidden');
            showError(scratchJdError, 'Something went wrong: ' + err.message);
        } finally {
            scratchSubmitBtn.disabled = false;
        }
    });

    function collectEntries(containerId) {
        const container = document.getElementById(containerId);
        const entries = [];
        container.querySelectorAll('textarea').forEach(ta => {
            if (ta.value.trim()) entries.push(ta.value.trim());
        });
        return entries;
    }

    // ── Shared API helper ───────────────────────────────────────────
    async function generateResume(formData) {
        const response = await fetch(`https://beula-doleritic-wispily.ngrok-free.dev/generate-resume`, {
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

        return await response.blob();
    }

    function autoDownload(blob) {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'tailored_resume.pdf';
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    }

    // ── ATS Score ────────────────────────────────────────────────────
    async function fetchATSScore(pdfBlob, jdTextValue) {
        try {
            const atsFormData = new FormData();
            atsFormData.append('jd_text', jdTextValue);
            atsFormData.append('resume_pdf', pdfBlob, 'tailored_resume.pdf');

            const atsResponse = await fetch(`https://beula-doleritic-wispily.ngrok-free.dev/ats-score`, {
                method: 'POST',
                body: atsFormData,
            });

            atsProgressCard.style.display = 'none';

            if (!atsResponse.ok) {
                downloadCard.style.display = 'flex';
                atsSection.style.display = 'none';
                return;
            }

            const data = await atsResponse.json();
            displayATSScore(data);
            downloadCard.style.display = 'flex';
            atsSection.style.display = 'block';
        } catch (err) {
            atsProgressCard.style.display = 'none';
            downloadCard.style.display = 'flex';
            atsSection.style.display = 'none';
            console.error('ATS scoring failed:', err);
        }
    }

    function displayATSScore(data) {
        const score = data.overall_score || 0;
        const circumference = 326.73;
        const offset = circumference - (score / 100) * circumference;

        atsRingFill.style.strokeDashoffset = circumference;
        requestAnimationFrame(() => {
            atsRingFill.style.transition = 'stroke-dashoffset 1.5s ease-out';
            atsRingFill.style.strokeDashoffset = offset;
        });

        let color;
        if (score >= 80) color = '#22c55e';
        else if (score >= 60) color = '#eab308';
        else color = '#ef4444';
        atsRingFill.style.stroke = color;

        animateNumber(atsScoreNumber, 0, score, 1500);

        const breakdown = data.breakdown || {};
        setBar('barKeyword', 'valKeyword', breakdown.keyword_match);
        setBar('barSkills', 'valSkills', breakdown.skills_coverage);
        setBar('barExperience', 'valExperience', breakdown.experience_relevance);
        setBar('barFormatting', 'valFormatting', breakdown.formatting_score);

        const suggestions = data.suggestions || [];
        if (suggestions.length > 0) {
            suggestionsList.innerHTML = '';
            suggestions.forEach(s => {
                const li = document.createElement('li');
                li.textContent = s;
                suggestionsList.appendChild(li);
            });
            atsSuggestions.style.display = 'block';
        } else {
            atsSuggestions.style.display = 'none';
        }
    }

    function setBar(barId, valId, value) {
        const bar = document.getElementById(barId);
        const val = document.getElementById(valId);
        const v = value || 0;
        bar.style.width = '0%';
        val.textContent = v + '%';
        requestAnimationFrame(() => {
            bar.style.transition = 'width 1.2s ease-out';
            bar.style.width = v + '%';
        });
        if (v >= 80) bar.style.background = '#22c55e';
        else if (v >= 60) bar.style.background = '#eab308';
        else bar.style.background = '#ef4444';
    }

    function animateNumber(el, from, to, duration) {
        const start = performance.now();
        function step(now) {
            const progress = Math.min((now - start) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            el.textContent = Math.round(from + (to - from) * eased);
            if (progress < 1) requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
    }

    // ── Regenerate ──────────────────────────────────────────────────
    regenerateBtn.addEventListener('click', () => {
        downloadCard.style.display = 'none';
        atsSection.style.display = 'none';
        atsSuggestions.style.display = 'none';
        atsRingFill.style.strokeDashoffset = 326.73;
        atsRingFill.style.transition = 'none';
        atsScoreNumber.textContent = '0';

        // Reset forms
        uploadForm.reset();
        scratchForm.reset();
        filePreview.style.display = 'none';
        dropZone.querySelector('.drop-content').style.display = '';

        // Remove dynamically added entries (keep first in each group)
        ['educationEntries', 'experienceEntries', 'projectEntries', 'achievementEntries'].forEach(id => {
            const container = document.getElementById(id);
            const entries = container.querySelectorAll('.dynamic-entry');
            entries.forEach((entry, i) => {
                if (i > 0) entry.remove();
            });
        });

        showView(viewLanding);
    });

    // ── Real-time validation ────────────────────────────────────────
    jdText.addEventListener('input', () => { if (jdText.value.trim()) clearError(jdError); });
    resumeFile.addEventListener('change', () => { if (resumeFile.files.length) clearError(fileError); });

    // ── Helpers ─────────────────────────────────────────────────────
    function showError(el, message) { el.textContent = message; }
    function clearError(el) { el.textContent = ''; }
});
