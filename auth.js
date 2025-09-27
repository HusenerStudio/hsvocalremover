// Authentication System for HSVocalRemover
class AuthSystem {
    constructor() {
        this.currentUser = null;
        this.users = JSON.parse(localStorage.getItem('hsvr_users')) || {};
        this.sessions = JSON.parse(localStorage.getItem('hsvr_sessions')) || {};
        this.init();
    }

    init() {
        // Check for existing session
        this.checkSession();
        
        // Initialize theme
        this.initTheme();
        
        // Initialize auth forms if on login page
        if (document.getElementById('login-form-element')) {
            this.initAuthForms();
        }
        
        // Initialize user menu
        this.initUserMenu();
    }

    // Theme functionality
    initTheme() {
        const themeToggleBtn = document.getElementById('theme-toggle');
        const body = document.body;

        // Load saved theme
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme) {
            body.classList.add(savedTheme);
            if (savedTheme === 'dark-theme') {
                themeToggleBtn.querySelector('i').classList.remove('fa-moon');
                themeToggleBtn.querySelector('i').classList.add('fa-sun');
            }
        }

        themeToggleBtn.addEventListener('click', () => {
            body.classList.toggle('dark-theme');
            if (body.classList.contains('dark-theme')) {
                localStorage.setItem('theme', 'dark-theme');
                themeToggleBtn.querySelector('i').classList.remove('fa-moon');
                themeToggleBtn.querySelector('i').classList.add('fa-sun');
            } else {
                localStorage.setItem('theme', 'light-theme');
                themeToggleBtn.querySelector('i').classList.remove('fa-sun');
                themeToggleBtn.querySelector('i').classList.add('fa-moon');
            }
        });
    }

    // Initialize authentication forms
    initAuthForms() {
        const authTabs = document.querySelectorAll('.auth-tab-button');
        const authForms = document.querySelectorAll('.auth-form-container');
        const loginForm = document.getElementById('login-form-element');
        const registerForm = document.getElementById('register-form-element');
        const passwordToggles = document.querySelectorAll('.password-toggle');

        // Tab switching
        authTabs.forEach((tab, index) => {
            tab.addEventListener('click', () => {
                authTabs.forEach(t => t.classList.remove('active'));
                authForms.forEach(f => f.classList.remove('active'));
                
                tab.classList.add('active');
                authForms[index].classList.add('active');
            });
        });

        // Password visibility toggles
        passwordToggles.forEach(toggle => {
            toggle.addEventListener('click', () => {
                const targetId = toggle.getAttribute('data-target');
                const targetInput = document.getElementById(targetId);
                const icon = toggle.querySelector('i');
                
                if (targetInput.type === 'password') {
                    targetInput.type = 'text';
                    icon.classList.remove('fa-eye');
                    icon.classList.add('fa-eye-slash');
                } else {
                    targetInput.type = 'password';
                    icon.classList.remove('fa-eye-slash');
                    icon.classList.add('fa-eye');
                }
            });
        });

        // Form submissions
        if (loginForm) {
            loginForm.addEventListener('submit', (e) => this.handleLogin(e));
        }
        
        if (registerForm) {
            registerForm.addEventListener('submit', (e) => this.handleRegister(e));
        }

        // Social login buttons (placeholder functionality)
        document.querySelectorAll('.social-button').forEach(button => {
            button.addEventListener('click', () => {
                this.showStatus('Social login coming soon!', 'info');
            });
        });
    }

    // Initialize user menu
    initUserMenu() {
        const userInfo = document.getElementById('user-info');
        const userDropdown = document.getElementById('user-dropdown');
        const userMenuToggle = document.getElementById('user-menu-toggle');
        const loginLink = document.getElementById('login-link');
        const logoutLink = document.getElementById('logout-link');
        const userName = document.getElementById('user-name');

        if (userMenuToggle) {
            userMenuToggle.addEventListener('click', () => {
                userDropdown.classList.toggle('show');
            });
        }

        if (logoutLink) {
            logoutLink.addEventListener('click', (e) => {
                e.preventDefault();
                this.logout();
            });
        }

        // Update UI based on login status
        this.updateUserUI();

        // Close dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (userDropdown && !userInfo.contains(e.target)) {
                userDropdown.classList.remove('show');
            }
        });
    }

    // Handle user registration
    async handleRegister(e) {
        e.preventDefault();
        
        const formData = new FormData(e.target);
        const name = formData.get('name').trim();
        const email = formData.get('email').trim().toLowerCase();
        const password = formData.get('password');
        const confirmPassword = formData.get('confirmPassword');

        // Validation
        if (!name || !email || !password) {
            this.showStatus('Please fill in all fields', 'error');
            return;
        }

        if (password !== confirmPassword) {
            this.showStatus('Passwords do not match', 'error');
            return;
        }

        if (password.length < 6) {
            this.showStatus('Password must be at least 6 characters long', 'error');
            return;
        }

        if (this.users[email]) {
            this.showStatus('An account with this email already exists', 'error');
            return;
        }

        // Create user
        const user = {
            id: Date.now().toString(),
            name,
            email,
            password: this.hashPassword(password),
            createdAt: new Date().toISOString(),
            history: []
        };

        this.users[email] = user;
        localStorage.setItem('hsvr_users', JSON.stringify(this.users));

        this.showStatus('Account created successfully! Please log in.', 'success');
        
        // Switch to login tab
        setTimeout(() => {
            document.querySelector('.auth-tab-button[data-tab="login"]').click();
        }, 1500);
    }

    // Handle user login
    async handleLogin(e) {
        e.preventDefault();
        
        const formData = new FormData(e.target);
        const email = formData.get('email').trim().toLowerCase();
        const password = formData.get('password');
        const rememberMe = document.getElementById('remember-me').checked;

        if (!email || !password) {
            this.showStatus('Please enter both email and password', 'error');
            return;
        }

        const user = this.users[email];
        if (!user || !this.verifyPassword(password, user.password)) {
            this.showStatus('Invalid email or password', 'error');
            return;
        }

        // Create session
        const sessionId = this.generateSessionId();
        const session = {
            userId: user.id,
            email: user.email,
            name: user.name,
            createdAt: new Date().toISOString(),
            rememberMe
        };

        this.sessions[sessionId] = session;
        localStorage.setItem('hsvr_sessions', JSON.stringify(this.sessions));
        
        if (rememberMe) {
            localStorage.setItem('hsvr_current_session', sessionId);
        } else {
            sessionStorage.setItem('hsvr_current_session', sessionId);
        }

        this.currentUser = session;
        this.showStatus('Login successful! Redirecting...', 'success');
        
        // Redirect to main page
        setTimeout(() => {
            window.location.href = 'index.html';
        }, 1500);
    }

    // Check for existing session
    checkSession() {
        const sessionId = localStorage.getItem('hsvr_current_session') || 
                         sessionStorage.getItem('hsvr_current_session');
        
        if (sessionId && this.sessions[sessionId]) {
            this.currentUser = this.sessions[sessionId];
            this.updateUserUI();
        }
    }

    // Update user interface based on login status
    updateUserUI() {
        const userName = document.getElementById('user-name');
        const loginLink = document.getElementById('login-link');
        const logoutLink = document.getElementById('logout-link');

        if (this.currentUser) {
            if (userName) userName.textContent = this.currentUser.name;
            if (loginLink) loginLink.classList.add('hidden');
            if (logoutLink) logoutLink.classList.remove('hidden');
        } else {
            if (userName) userName.textContent = 'Guest';
            if (loginLink) loginLink.classList.remove('hidden');
            if (logoutLink) logoutLink.classList.add('hidden');
        }
    }

    // Logout user
    logout() {
        const sessionId = localStorage.getItem('hsvr_current_session') || 
                         sessionStorage.getItem('hsvr_current_session');
        
        if (sessionId) {
            delete this.sessions[sessionId];
            localStorage.setItem('hsvr_sessions', JSON.stringify(this.sessions));
            localStorage.removeItem('hsvr_current_session');
            sessionStorage.removeItem('hsvr_current_session');
        }

        this.currentUser = null;
        this.updateUserUI();
        
        // Redirect to home if on protected page
        if (window.location.pathname.includes('history.html')) {
            window.location.href = 'index.html';
        }
    }

    // Save user history item
    saveToHistory(item) {
        if (!this.currentUser) return false;

        const user = this.users[this.currentUser.email];
        if (!user) return false;

        const historyItem = {
            id: Date.now().toString(),
            timestamp: new Date().toISOString(),
            ...item
        };

        user.history.unshift(historyItem);
        
        // Keep only last 100 items
        if (user.history.length > 100) {
            user.history = user.history.slice(0, 100);
        }

        this.users[this.currentUser.email] = user;
        localStorage.setItem('hsvr_users', JSON.stringify(this.users));
        
        return true;
    }

    // Get user history
    getHistory() {
        if (!this.currentUser) return [];
        
        const user = this.users[this.currentUser.email];
        return user ? user.history : [];
    }

    // Utility functions
    hashPassword(password) {
        // Simple hash for demo - in production, use proper hashing
        let hash = 0;
        for (let i = 0; i < password.length; i++) {
            const char = password.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return hash.toString();
    }

    verifyPassword(password, hash) {
        return this.hashPassword(password) === hash;
    }

    generateSessionId() {
        return Date.now().toString() + Math.random().toString(36).substr(2, 9);
    }

    showStatus(message, type = 'info') {
        const statusElement = document.getElementById('auth-status');
        if (!statusElement) return;

        statusElement.textContent = message;
        statusElement.className = `auth-status ${type}`;
        statusElement.style.display = 'block';

        setTimeout(() => {
            statusElement.style.display = 'none';
        }, 5000);
    }

    // Check if user is logged in
    isLoggedIn() {
        return this.currentUser !== null;
    }

    // Get current user
    getCurrentUser() {
        return this.currentUser;
    }
}

// Initialize authentication system
document.addEventListener('DOMContentLoaded', () => {
    window.authSystem = new AuthSystem();
});