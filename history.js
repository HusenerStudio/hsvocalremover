// History Management System for HSVocalRemover
class HistoryManager {
    constructor() {
        this.currentPage = 1;
        this.itemsPerPage = 10;
        this.currentFilter = 'all';
        this.searchQuery = '';
        this.history = [];
        this.init();
    }

    init() {
        // Check if user is logged in
        if (!window.authSystem || !window.authSystem.isLoggedIn()) {
            this.redirectToLogin();
            return;
        }

        this.loadHistory();
        this.initializeEventListeners();
        this.renderHistory();
        this.updateStats();
    }

    redirectToLogin() {
        window.location.href = 'login.html';
    }

    loadHistory() {
        this.history = window.authSystem.getHistory();
    }

    initializeEventListeners() {
        // Search functionality
        const searchInput = document.getElementById('search-input');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.searchQuery = e.target.value.toLowerCase();
                this.currentPage = 1;
                this.renderHistory();
            });
        }

        // Filter functionality
        const filterSelect = document.getElementById('filter-select');
        if (filterSelect) {
            filterSelect.addEventListener('change', (e) => {
                this.currentFilter = e.target.value;
                this.currentPage = 1;
                this.renderHistory();
            });
        }

        // Clear history button
        const clearHistoryBtn = document.getElementById('clear-history-btn');
        if (clearHistoryBtn) {
            clearHistoryBtn.addEventListener('click', () => {
                this.clearHistory();
            });
        }

        // Pagination buttons
        const prevPageBtn = document.getElementById('prev-page');
        const nextPageBtn = document.getElementById('next-page');
        
        if (prevPageBtn) {
            prevPageBtn.addEventListener('click', () => {
                if (this.currentPage > 1) {
                    this.currentPage--;
                    this.renderHistory();
                }
            });
        }

        if (nextPageBtn) {
            nextPageBtn.addEventListener('click', () => {
                const filteredHistory = this.getFilteredHistory();
                const totalPages = Math.ceil(filteredHistory.length / this.itemsPerPage);
                if (this.currentPage < totalPages) {
                    this.currentPage++;
                    this.renderHistory();
                }
            });
        }
    }

    getFilteredHistory() {
        let filtered = this.history;

        // Apply search filter
        if (this.searchQuery) {
            filtered = filtered.filter(item => 
                item.fileName?.toLowerCase().includes(this.searchQuery) ||
                item.type?.toLowerCase().includes(this.searchQuery) ||
                item.modelName?.toLowerCase().includes(this.searchQuery)
            );
        }

        // Apply type filter
        if (this.currentFilter !== 'all') {
            filtered = filtered.filter(item => item.type === this.currentFilter);
        }

        return filtered;
    }

    renderHistory() {
        const historyContainer = document.getElementById('history-items');
        const emptyMessage = document.getElementById('empty-history');
        const paginationInfo = document.getElementById('pagination-info');
        
        if (!historyContainer) return;

        const filteredHistory = this.getFilteredHistory();
        
        if (filteredHistory.length === 0) {
            historyContainer.innerHTML = '';
            if (emptyMessage) emptyMessage.style.display = 'block';
            this.updatePagination(0, 0);
            return;
        }

        if (emptyMessage) emptyMessage.style.display = 'none';

        // Calculate pagination
        const startIndex = (this.currentPage - 1) * this.itemsPerPage;
        const endIndex = startIndex + this.itemsPerPage;
        const pageItems = filteredHistory.slice(startIndex, endIndex);
        const totalPages = Math.ceil(filteredHistory.length / this.itemsPerPage);

        // Render items
        historyContainer.innerHTML = pageItems.map(item => this.renderHistoryItem(item)).join('');

        // Update pagination
        this.updatePagination(filteredHistory.length, totalPages);

        // Add event listeners to action buttons
        this.addItemEventListeners();
    }

    renderHistoryItem(item) {
        const date = new Date(item.timestamp).toLocaleString();
        const typeIcon = this.getTypeIcon(item.type);
        const statusBadge = this.getStatusBadge(item.status);
        
        return `
            <div class="history-item" data-id="${item.id}">
                <div class="item-icon">
                    <i class="${typeIcon}"></i>
                </div>
                <div class="item-details">
                    <div class="item-header">
                        <h3 class="item-title">${item.fileName || 'Unknown File'}</h3>
                        ${statusBadge}
                    </div>
                    <div class="item-meta">
                        <span class="item-type">${this.formatType(item.type)}</span>
                        <span class="item-date">${date}</span>
                        ${item.modelName ? `<span class="item-model">Model: ${item.modelName}</span>` : ''}
                    </div>
                    ${item.description ? `<p class="item-description">${item.description}</p>` : ''}
                </div>
                <div class="item-actions">
                    ${item.downloadUrl ? `<button class="action-btn download-btn" data-action="download" data-url="${item.downloadUrl}">
                        <i class="fas fa-download"></i>
                    </button>` : ''}
                    ${item.playUrl ? `<button class="action-btn play-btn" data-action="play" data-url="${item.playUrl}">
                        <i class="fas fa-play"></i>
                    </button>` : ''}
                    <button class="action-btn delete-btn" data-action="delete" data-id="${item.id}">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        `;
    }

    getTypeIcon(type) {
        const icons = {
            'upload': 'fas fa-upload',
            'separation': 'fas fa-cut',
            'training': 'fas fa-brain',
            'download': 'fas fa-download'
        };
        return icons[type] || 'fas fa-file';
    }

    getStatusBadge(status) {
        if (!status) return '';
        
        const badges = {
            'completed': '<span class="status-badge success">Completed</span>',
            'processing': '<span class="status-badge processing">Processing</span>',
            'failed': '<span class="status-badge error">Failed</span>',
            'pending': '<span class="status-badge pending">Pending</span>'
        };
        
        return badges[status] || '';
    }

    formatType(type) {
        const types = {
            'upload': 'File Upload',
            'separation': 'Vocal Separation',
            'training': 'Model Training',
            'download': 'Download'
        };
        return types[type] || type;
    }

    addItemEventListeners() {
        const actionButtons = document.querySelectorAll('.action-btn');
        
        actionButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.stopPropagation();
                const action = button.getAttribute('data-action');
                const id = button.getAttribute('data-id');
                const url = button.getAttribute('data-url');
                
                switch (action) {
                    case 'download':
                        this.downloadItem(url);
                        break;
                    case 'play':
                        this.playItem(url);
                        break;
                    case 'delete':
                        this.deleteItem(id);
                        break;
                }
            });
        });
    }

    downloadItem(url) {
        if (!url) return;
        
        const link = document.createElement('a');
        link.href = url;
        link.download = '';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    playItem(url) {
        if (!url) return;
        
        // Create or update audio player
        let audioPlayer = document.getElementById('history-audio-player');
        if (!audioPlayer) {
            audioPlayer = document.createElement('audio');
            audioPlayer.id = 'history-audio-player';
            audioPlayer.controls = true;
            audioPlayer.style.position = 'fixed';
            audioPlayer.style.bottom = '20px';
            audioPlayer.style.right = '20px';
            audioPlayer.style.zIndex = '1000';
            document.body.appendChild(audioPlayer);
        }
        
        audioPlayer.src = url;
        audioPlayer.play();
    }

    deleteItem(id) {
        if (!confirm('Are you sure you want to delete this item from your history?')) {
            return;
        }
        
        // Remove from current user's history
        const user = window.authSystem.users[window.authSystem.currentUser.email];
        if (user) {
            user.history = user.history.filter(item => item.id !== id);
            window.authSystem.users[window.authSystem.currentUser.email] = user;
            localStorage.setItem('hsvr_users', JSON.stringify(window.authSystem.users));
        }
        
        // Reload and re-render
        this.loadHistory();
        this.renderHistory();
        this.updateStats();
        
        this.showStatus('Item deleted successfully', 'success');
    }

    clearHistory() {
        if (!confirm('Are you sure you want to clear all history? This action cannot be undone.')) {
            return;
        }
        
        // Clear current user's history
        const user = window.authSystem.users[window.authSystem.currentUser.email];
        if (user) {
            user.history = [];
            window.authSystem.users[window.authSystem.currentUser.email] = user;
            localStorage.setItem('hsvr_users', JSON.stringify(window.authSystem.users));
        }
        
        // Reload and re-render
        this.loadHistory();
        this.renderHistory();
        this.updateStats();
        
        this.showStatus('History cleared successfully', 'success');
    }

    updatePagination(totalItems, totalPages) {
        const paginationInfo = document.getElementById('pagination-info');
        const prevPageBtn = document.getElementById('prev-page');
        const nextPageBtn = document.getElementById('next-page');
        
        if (paginationInfo) {
            if (totalItems === 0) {
                paginationInfo.textContent = 'No items found';
            } else {
                const startItem = (this.currentPage - 1) * this.itemsPerPage + 1;
                const endItem = Math.min(this.currentPage * this.itemsPerPage, totalItems);
                paginationInfo.textContent = `Showing ${startItem}-${endItem} of ${totalItems} items (Page ${this.currentPage} of ${totalPages})`;
            }
        }
        
        if (prevPageBtn) {
            prevPageBtn.disabled = this.currentPage <= 1;
        }
        
        if (nextPageBtn) {
            nextPageBtn.disabled = this.currentPage >= totalPages;
        }
    }

    updateStats() {
        const totalItems = document.getElementById('total-items');
        const totalUploads = document.getElementById('total-uploads');
        const totalSeparations = document.getElementById('total-separations');
        
        if (totalItems) totalItems.textContent = this.history.length;
        
        if (totalUploads) {
            const uploads = this.history.filter(item => item.type === 'upload').length;
            totalUploads.textContent = uploads;
        }
        
        if (totalSeparations) {
            const separations = this.history.filter(item => item.type === 'separation').length;
            totalSeparations.textContent = separations;
        }
    }

    showStatus(message, type = 'info') {
        // Create status message element if it doesn't exist
        let statusElement = document.getElementById('history-status');
        if (!statusElement) {
            statusElement = document.createElement('div');
            statusElement.id = 'history-status';
            statusElement.className = 'status-message';
            document.body.appendChild(statusElement);
        }

        statusElement.textContent = message;
        statusElement.className = `status-message ${type}`;
        statusElement.style.display = 'block';

        setTimeout(() => {
            statusElement.style.display = 'none';
        }, 3000);
    }

    // Public method to add new history item
    static addHistoryItem(item) {
        if (window.authSystem && window.authSystem.isLoggedIn()) {
            window.authSystem.saveToHistory(item);
        }
    }
}

// Initialize history manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    if (window.location.pathname.includes('history.html')) {
        window.historyManager = new HistoryManager();
    }
});

// Export for use in other scripts
window.HistoryManager = HistoryManager;