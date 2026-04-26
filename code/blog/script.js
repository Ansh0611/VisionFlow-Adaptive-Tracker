/* ═══════════════════════════════════════════════════════════
   VisionFlow Blog — Application Logic
   Client-side Markdown rendering with marked.js + hljs
   ═══════════════════════════════════════════════════════════ */

// ── Author Metadata ─────────────────────────────────────────
const AUTHORS = {
    'lakshit-mehta': {
        name: 'Lakshit Mehta',
        role: 'Algorithmic Architecture & Optimization',
        initials: 'LM',
        avatarClass: 'avatar-blue',
        glowClass: 'glow-blue',
        title: 'The Mathematics of Pursuit — Hungarian Assignment & Kalman Prediction in Multi-Object Tracking',
        date: 'April 2026',
        file: 'posts/lakshit-mehta.md'
    },
    'siddhi-pateriya': {
        name: 'Siddhi Pateriya',
        role: 'Geometric Vision & Pose Estimation',
        initials: 'SP',
        avatarClass: 'avatar-purple',
        glowClass: 'glow-purple',
        title: 'Reconstructing the Unseen — How Visual Odometry Recovers Camera Trajectory from 2D Images',
        date: 'April 2026',
        file: 'posts/siddhi-pateriya.md'
    },
    'chirag-taneja': {
        name: 'Chirag Taneja',
        role: 'System Design & Adaptive Architecture',
        initials: 'CT',
        avatarClass: 'avatar-cyan',
        glowClass: 'glow-cyan',
        title: 'One System, Two Worlds — Designing an Adaptive Vision Architecture That Thinks Before It Sees',
        date: 'April 2026',
        file: 'posts/chirag-taneja.md'
    },
    'ansh-sidana': {
        name: 'Ansh Sidana',
        role: 'Motion Analysis & Scene Understanding',
        initials: 'AS',
        avatarClass: 'avatar-green',
        glowClass: 'glow-green',
        title: 'The First Decision — How Optical Flow and Background Modeling Classify the Visual World',
        date: 'April 2026',
        file: 'posts/ansh-sidana.md'
    }
};

// ── DOM References ──────────────────────────────────────────
const homeSection = document.getElementById('home-section');
const postSection = document.getElementById('post-section');
const postHeader = document.getElementById('post-header');
const postBody = document.getElementById('post-body');
const backBtn = document.getElementById('back-btn');
const navbar = document.getElementById('navbar');
const navLinks = document.querySelectorAll('.nav-link');
const mobileToggle = document.getElementById('mobile-toggle');
const navLinksContainer = document.getElementById('nav-links');

// ── Markdown Configuration ──────────────────────────────────
marked.setOptions({
    breaks: true,
    gfm: true,
    highlight: function(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
    }
});

// ── Navigation ──────────────────────────────────────────────
function setActiveNav(key) {
    navLinks.forEach(link => link.classList.remove('active'));
    if (key === 'home') {
        document.querySelector('[data-section="home"]')?.classList.add('active');
    } else {
        document.querySelector(`.nav-link[data-author="${key}"]`)?.classList.add('active');
    }
}

function showHome() {
    postSection.classList.add('hidden');
    homeSection.style.display = '';
    setActiveNav('home');
    window.scrollTo({ top: 0, behavior: 'smooth' });
    // Close mobile menu
    navLinksContainer.classList.remove('open');
}

async function showPost(authorKey) {
    const author = AUTHORS[authorKey];
    if (!author) return;

    // Build post header
    postHeader.innerHTML = `
        <div class="post-author-info">
            <div class="post-avatar ${author.avatarClass}">${author.initials}</div>
            <div>
                <div class="post-author-name">${author.name}</div>
                <div class="post-author-role">${author.role}</div>
            </div>
        </div>
        <h1 class="post-title">${author.title}</h1>
        <div class="post-date">${author.date} · VisionFlow Adaptive Tracker</div>
    `;

    // Fetch and render markdown
    postBody.innerHTML = '<p style="color: var(--text-muted);">Loading post…</p>';

    try {
        const response = await fetch(author.file);
        if (!response.ok) throw new Error('Failed to load');
        const markdown = await response.text();
        postBody.innerHTML = marked.parse(markdown);
        
        // Apply syntax highlighting to any code blocks
        postBody.querySelectorAll('pre code').forEach(block => {
            hljs.highlightElement(block);
        });
    } catch (err) {
        postBody.innerHTML = `<p style="color: var(--accent-pink);">Could not load the blog post. Please ensure you're running this from a web server.</p>`;
    }

    // Switch view
    homeSection.style.display = 'none';
    postSection.classList.remove('hidden');
    setActiveNav(authorKey);
    window.scrollTo({ top: 0, behavior: 'smooth' });

    // Close mobile menu
    navLinksContainer.classList.remove('open');
}

// ── Event Listeners ─────────────────────────────────────────

// Nav links
navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        if (link.dataset.section === 'home') {
            showHome();
        } else if (link.dataset.author) {
            showPost(link.dataset.author);
        }
    });
});

// Logo click → home
document.getElementById('nav-logo-link').addEventListener('click', (e) => {
    e.preventDefault();
    showHome();
});

// Back button
backBtn.addEventListener('click', () => showHome());

// Author cards (click anywhere on card)
document.querySelectorAll('.author-card').forEach(card => {
    card.addEventListener('click', () => {
        const authorKey = card.dataset.author;
        if (authorKey) showPost(authorKey);
    });
});

// "Read Insight" buttons (prevent double-fire from card click)
document.querySelectorAll('.read-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const card = btn.closest('.author-card');
        if (card?.dataset.author) showPost(card.dataset.author);
    });
});

// Mobile menu toggle
mobileToggle.addEventListener('click', () => {
    navLinksContainer.classList.toggle('open');
});

// Scroll effect for navbar
window.addEventListener('scroll', () => {
    if (window.scrollY > 20) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
});

// ── Card Entrance Animation ─────────────────────────────────
const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry, index) => {
        if (entry.isIntersecting) {
            entry.target.style.animationDelay = `${index * 0.1}s`;
            entry.target.style.animation = 'fadeInUp 0.5s ease-out forwards';
            observer.unobserve(entry.target);
        }
    });
}, { threshold: 0.1 });

document.querySelectorAll('.author-card').forEach(card => {
    card.style.opacity = '0';
    observer.observe(card);
});
