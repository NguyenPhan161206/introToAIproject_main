const ROWS = 9;
const COLS = 9;

const state = {
    board: Array.from({ length: ROWS }, () => Array(COLS).fill(0)),
    gameOver: false,
    winner: null,
    mode: 'pvai_trad',
    turn: 'player',
    depth: 3,
    scores: { player: 0, ai: 0, draw: 0 },
    moveCount: 0,
    thinking: false,
    lastMove: null,
    annAvailable: false
};

const boardEl = document.getElementById('board');
const gameStatusEl = document.getElementById('gameStatus');
const playerScoreEl = document.getElementById('playerScore');
const aiScoreEl = document.getElementById('aiScore');
const drawScoreEl = document.getElementById('drawScore');
const thinkingTimeEl = document.getElementById('thinkingTime');
const moveCountEl = document.getElementById('moveCount');
const thinkingIndicatorEl = document.getElementById('thinkingIndicator');
const depthSlider = document.getElementById('depthSlider');
const depthValueEl = document.getElementById('depthValue');
const newGameBtn = document.getElementById('newGameBtn');
const aiVsAiBtn = document.getElementById('aiVsAiBtn');
const modePvaiAnnBtn = document.getElementById('modePvaiAnn');
const modePvaiTradBtn = document.getElementById('modePvaiTrad');
const modePvpBtn = document.getElementById('modePvp');
const aiControlsEl = document.getElementById('aiControls');
const scoreLabel1El = document.getElementById('scoreLabel1');
const scoreLabel2El = document.getElementById('scoreLabel2');
const errorMsgEl = document.getElementById('errorMessage');

let errorTimeout = null;


function showError(message) {
    if (!errorMsgEl) return;
    errorMsgEl.textContent = message;
    errorMsgEl.style.display = 'block';
    if (errorTimeout) clearTimeout(errorTimeout);
    errorTimeout = setTimeout(() => {
        errorMsgEl.style.display = 'none';
    }, 3000);
}


function createBoard() {
    boardEl.innerHTML = '';
    boardEl.className = 'board';

    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.dataset.row = r;
            cell.dataset.col = c;

            const lineH = document.createElement('div');
            lineH.className = 'grid-line-h';
            cell.appendChild(lineH);

            const lineV = document.createElement('div');
            lineV.className = 'grid-line-v';
            cell.appendChild(lineV);

            cell.addEventListener('click', () => handleCellClick(r, c));
            boardEl.appendChild(cell);
        }
    }

    updateBoard();
}

function getCell(row, col) {
    return boardEl.querySelector(`.cell[data-row="${row}"][data-col="${col}"]`);
}

function updateBoard() {
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const cell = getCell(r, c);
            if (!cell) continue;

            const existingStone = cell.querySelector('.stone');
            if (existingStone) existingStone.remove();

            const val = state.board[r][c];
            if (val !== 0) {
                const stone = document.createElement('div');
                stone.className = `stone ${val === 1 ? 'black' : 'white'}`;
                if (state.lastMove && state.lastMove[0] === r && state.lastMove[1] === c) {
                    stone.classList.add('last-move');
                }
                cell.appendChild(stone);
            }
        }
    }

    boardEl.classList.toggle('game-over', state.gameOver);
    boardEl.classList.toggle('thinking', state.thinking);
}

function setGameStatus(message, type) {
    gameStatusEl.textContent = message;
    gameStatusEl.className = 'game-status ' + type;
}

function updateScoreboard() {
    playerScoreEl.textContent = state.scores.player;
    aiScoreEl.textContent = state.scores.ai;
    drawScoreEl.textContent = state.scores.draw;
}

function setThinking(active) {
    state.thinking = active;
    thinkingIndicatorEl.classList.toggle('active', active);
    boardEl.classList.toggle('thinking', active);
}


function updateModeUI() {
    const isPvP = state.mode === 'pvp';
    const isAnn = state.mode === 'pvai_ann';

    aiControlsEl?.classList.toggle('hidden', isPvP);
    aiVsAiBtn?.classList.toggle('hidden', isPvP);
    modePvaiAnnBtn?.classList.toggle('active', isAnn);
    modePvaiTradBtn?.classList.toggle('active', !isPvP && !isAnn);
    modePvpBtn?.classList.toggle('active', isPvP);

    if (isPvP) {
        scoreLabel1El.textContent = 'ĐEN';
        scoreLabel2El.textContent = 'TRẮNG';
    } else {
        scoreLabel1El.textContent = 'BẠN (Đen)';
        scoreLabel2El.textContent = 'AI (Trắng)';
    }

    if (!state.gameOver) {
        updateTurnUI();
    }
}

function updateTurnUI() {
    if (state.mode === 'pvp') {
        const turnNum = state.turn === 'player' || state.turn === 1 ? 1 : 2;
        const label = turnNum === 1 ? 'Đen' : 'Trắng';
        setGameStatus(`Lượt ${label}`, 'playing');
    } else {
        if (state.turn === 'player') {
            setGameStatus('Lượt của bạn', 'playing');
        } else if (state.turn === 'ai') {
            setGameStatus('AI đang suy nghĩ...', 'playing');
        }
    }
}


async function handleCellClick(row, col) {
    if (state.gameOver || state.thinking) return;
    if (state.board[row][col] !== 0) return;

    if (state.mode.startsWith('pvai') && state.turn !== 'player') {
        showError('Đang đến lượt AI');
        return;
    }

    setThinking(true);

    try {
        const res = await fetch('/api/player_move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ row, col })
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            showError(errData.error || 'Nước đi không hợp lệ');
            setThinking(false);
            return;
        }

        const data = await res.json();
        applyBoardData(data);

        if (data.game_over) {
            state.turn = 'none';
            handleGameOver(data);
            setThinking(false);
        } else if (state.mode.startsWith('pvai')) {
            state.turn = 'ai';
            updateTurnUI();
            setThinking(false);
            await makeAIMove();
        } else {
            state.turn = data.current_turn;
            updateTurnUI();
            setThinking(false);
        }
    } catch (err) {
        console.error('Error:', err);
        showError('Lỗi kết nối đến server');
        setThinking(false);
    }
}

async function makeAIMove() {
    setThinking(true);
    state.turn = 'ai';
    updateTurnUI();

    try {
        const res = await fetch('/api/ai_move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ depth: state.depth })
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            showError(errData.error || 'AI gặp lỗi');
            state.turn = 'player';
            updateTurnUI();
            setThinking(false);
            return;
        }

        const data = await res.json();

        if (data.thinking_time_ms) {
            thinkingTimeEl.textContent = data.thinking_time_ms + 'ms';
        }

        applyBoardData(data);

        if (data.game_over) {
            state.turn = 'none';
            handleGameOver(data);
        } else {
            state.turn = 'player';
            updateTurnUI();
        }
    } catch (err) {
        console.error('Error:', err);
        showError('Lỗi kết nối khi AI suy nghĩ');
        state.turn = 'player';
        updateTurnUI();
    } finally {
        setThinking(false);
    }
}

function applyBoardData(data) {
    state.board = data.board;
    state.gameOver = data.game_over;
    state.winner = data.winner;

    if (data.last_move) {
        state.lastMove = [data.last_move[0], data.last_move[1]];
    }

    state.moveCount = 0;
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            if (data.board[r][c] !== 0) state.moveCount++;
        }
    }
    moveCountEl.textContent = state.moveCount;

    updateBoard();
}

function handleGameOver(data) {
    const w = data.winner || state.winner;

    if (state.mode === 'pvp') {
        if (w === 'black') {
            setGameStatus('Đen thắng!', 'player-wins');
            state.scores.player++;
        } else if (w === 'white') {
            setGameStatus('Trắng thắng!', 'ai-wins');
            state.scores.ai++;
        } else if (w === 'draw') {
            setGameStatus('Hòa!', 'draw');
            state.scores.draw++;
        }
    } else {
        if (w === 'player') {
            setGameStatus('Bạn thắng!', 'player-wins');
            state.scores.player++;
        } else if (w === 'ai') {
            setGameStatus('AI thắng!', 'ai-wins');
            state.scores.ai++;
        } else if (w === 'draw') {
            setGameStatus('Hòa!', 'draw');
            state.scores.draw++;
        }
    }
    updateScoreboard();
}

async function resetGame() {
    setThinking(true);

    try {
        const res = await fetch('/api/reset', { method: 'POST' });
        if (!res.ok) {
            showError('Không thể reset game');
            setThinking(false);
            return;
        }

        state.board = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
        state.gameOver = false;
        state.winner = null;
        state.moveCount = 0;
        state.lastMove = null;
        thinkingTimeEl.textContent = '—';
        moveCountEl.textContent = '0';

        state.turn = 'player';
        updateBoard();
        updateTurnUI();

        await fetchMode();
    } catch (err) {
        console.error('Error:', err);
        showError('Lỗi kết nối khi reset');
    } finally {
        setThinking(false);
    }
}

async function runAiVsAi() {
    setThinking(true);
    setGameStatus('AI đang tự đấu...', 'playing');

    try {
        const h1 = state.annAvailable ? 'ann' : 'traditional';
        const h2 = 'traditional';

        const res = await fetch('/api/ai_vs_ai', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                heuristic1: h1,
                heuristic2: h2,
                depth: state.depth
            })
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            showError(errData.error || 'AI vs AI thất bại');
            setThinking(false);
            return;
        }

        const data = await res.json();

        // Reset backend state to avoid desync
        const resetRes = await fetch('/api/reset', { method: 'POST' });
        if (!resetRes.ok) {
            showError('Lỗi đồng bộ sau AI vs AI');
        }

        state.board = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
        state.lastMove = null;

        for (const m of data.moves) {
            state.board[m.row][m.col] = m.player === 'black' ? 1 : 2;
            state.lastMove = [m.row, m.col];
        }

        if (data.moves.length > 0) {
            const last = data.moves[data.moves.length - 1];
            state.lastMove = [last.row, last.col];
        }

        state.gameOver = true;
        state.moveCount = data.moves.length;
        moveCountEl.textContent = state.moveCount;
        state.turn = 'none';
        updateBoard();

        if (data.winner === 'black') {
            setGameStatus('ANN thắng!', 'player-wins');
            state.scores.player++;
        } else if (data.winner === 'white') {
            setGameStatus('Truyền thống thắng!', 'ai-wins');
            state.scores.ai++;
        } else {
            setGameStatus('Hòa!', 'draw');
            state.scores.draw++;
        }
        updateScoreboard();
    } catch (err) {
        console.error('Error:', err);
        showError('Lỗi kết nối khi chạy AI vs AI');
    } finally {
        setThinking(false);
    }
}


async function setMode(newMode) {
    try {
        const res = await fetch('/api/set_mode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: newMode })
        });
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            showError(errData.error || 'Không thể đổi chế độ');
            return;
        }
        state.mode = newMode;
        updateModeUI();
        if (!state.gameOver) {
            updateTurnUI();
        }
    } catch (err) {
        console.error('Error:', err);
        showError('Lỗi kết nối khi đổi chế độ');
    }
}

async function fetchMode() {
    try {
        const res = await fetch('/api/mode');
        if (res.ok) {
            const data = await res.json();
            state.mode = data.mode;
            state.annAvailable = data.ann_available;
            updateModeUI();
        }
    } catch (err) {
        console.error('Error fetching mode:', err);
    }
}


depthSlider?.addEventListener('input', (e) => {
    state.depth = parseInt(e.target.value);
    depthValueEl.textContent = state.depth;
});

newGameBtn?.addEventListener('click', resetGame);
aiVsAiBtn?.addEventListener('click', runAiVsAi);
modePvaiAnnBtn?.addEventListener('click', () => setMode('pvai_ann'));
modePvaiTradBtn?.addEventListener('click', () => setMode('pvai_trad'));
modePvpBtn?.addEventListener('click', () => setMode('pvp'));

createBoard();
resetGame();
