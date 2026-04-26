const AUTHORS = {
    'lakshit-mehta': { file: 'posts/lakshit-mehta.md' },
    'siddhi-pateriya': { file: 'posts/siddhi-pateriya.md' },
    'chirag-taneja': { file: 'posts/chirag-taneja.md' },
    'ansh-sidana': { file: 'posts/ansh-sidana.md' }
};

const homeView = document.getElementById('home-view');
const postView = document.getElementById('post-view');
const postContent = document.getElementById('post-content');

document.getElementById('home-link').addEventListener('click', (e) => {
    e.preventDefault();
    postView.classList.add('hidden');
    homeView.classList.remove('hidden');
});

document.getElementById('back-btn').addEventListener('click', () => {
    postView.classList.add('hidden');
    homeView.classList.remove('hidden');
});

document.querySelectorAll('.post-link').forEach(link => {
    link.addEventListener('click', async (e) => {
        e.preventDefault();
        const author = link.dataset.author;
        postContent.innerHTML = '<p>Loading...</p>';
        homeView.classList.add('hidden');
        postView.classList.remove('hidden');

        try {
            const res = await fetch(AUTHORS[author].file);
            if (!res.ok) throw new Error('Network error');
            const md = await res.text();
            postContent.innerHTML = marked.parse(md);
            window.scrollTo(0, 0);
        } catch (err) {
            postContent.innerHTML = '<p>Error loading post. Ensure you are running this from a local web server (not file://).</p>';
        }
    });
});
