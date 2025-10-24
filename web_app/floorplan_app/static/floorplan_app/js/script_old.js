// FloorplanGenerator MTV behavior
(function(){
  const LDK_LEVELS = [1, 2, 3, 4];
  const TOILET_OPTIONS = [1, 2, 3];
  const ROOM_LABELS = {
    'Japan Style Room': 'guest room',
    'Study Room': 'study room',
    'Storage': 'storage',
    'Balcony': 'balcony'
  };
  const POLL_INTERVAL_MS = 3000;
  const POLL_TIMEOUT_MS = 5 * 60 * 1000;

  const unitEl = document.getElementById('unitBtns');
  const floorEl = document.getElementById('floorBtns');
  const totalAreaInput = document.getElementById('totalArea');
  const ldkSelect = document.getElementById('ldkSelect');
  const toiletSelect = document.getElementById('toiletSelect');
  const roomInputs = Array.from(document.querySelectorAll('.room-grid input[type="checkbox"]'));
  const unitLabel = document.getElementById('unitLabel');
  const generateBtn = document.querySelector('.generate-btn');
  const statusEl = document.createElement('div');
  statusEl.className = 'status-message status-info';
  if(generateBtn){
    generateBtn.insertAdjacentElement('afterend', statusEl);
  }

  const loadingOverlay = document.getElementById('loadingOverlay');
  const loadingMessageEl = loadingOverlay ? loadingOverlay.querySelector('.loading-message') : null;

  let pollTimer = null;
  let activeJobId = null;

  const state = {
    unit: unitEl ? unitEl.querySelector('.btn-toggle.active')?.dataset.unit || 'metric' : 'metric',
    floor: floorEl ? Number(floorEl.querySelector('.btn-toggle.active')?.dataset.floor || 1) : 1,
    totalArea: totalAreaInput ? Number(totalAreaInput.value) || 0 : 0,
    ldk: ldkSelect ? Number(ldkSelect.value) || 1 : 1,
    rooms: {},
    toiletCount: toiletSelect ? Number(toiletSelect.value) || 1 : 1
  };

  function getRoomName(input){
    return input.parentElement.textContent.trim();
  }

  function initialiseRooms(){
    roomInputs.forEach(input => {
      const roomName = getRoomName(input);
      state.rooms[roomName] = input.checked;
    });
  }

  function syncUnitButtons(){
    if(!unitEl) return;
    unitEl.querySelectorAll('.btn-toggle').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.unit === state.unit);
    });
    if(unitLabel) unitLabel.textContent = 'm²';
  }

  function syncFloorButtons(){
    if(!floorEl) return;
    floorEl.querySelectorAll('.btn-toggle').forEach(btn => {
      const floor = Number(btn.dataset.floor || 0);
      btn.classList.toggle('active', floor === state.floor);
    });
  }

  function syncFields(){
    if(totalAreaInput){
      totalAreaInput.value = state.totalArea ? String(state.totalArea) : '';
    }
    if(ldkSelect){
      ldkSelect.value = String(state.ldk);
    }
    if(toiletSelect){
      toiletSelect.value = String(state.toiletCount);
    }
    roomInputs.forEach(input => {
      const roomName = getRoomName(input);
      if(roomName in state.rooms){
        input.checked = state.rooms[roomName];
      }
    });
  }

  function clearPollTimer(){
    if(pollTimer){
      clearTimeout(pollTimer);
      pollTimer = null;
    }
  }

  function setStatus(message, type = 'info'){
    if(!statusEl) return;
    statusEl.textContent = message;
    statusEl.classList.remove('status-info', 'status-success', 'status-error');
    statusEl.classList.add(`status-${type}`);
  }

  function updateLoadingMessage(message){
    if(!loadingMessageEl || !message) return;
    loadingMessageEl.textContent = message;
  }

  function showLoadingOverlay(message){
    if(!loadingOverlay) return;
    if(message){
      updateLoadingMessage(message);
    }
    loadingOverlay.hidden = false;
    document.body.classList.add('overlay-active');
  }

  function hideLoadingOverlay(){
    if(loadingOverlay){
      loadingOverlay.hidden = true;
    }
    document.body.classList.remove('overlay-active');
  }

  function syncAll(){
    syncUnitButtons();
    syncFloorButtons();
    syncFields();
  }

  function formatCount(count, label){
    return `${count} ${label}`;
  }

  function buildRoomDescription(){
    const items = [];

    // LDK-derived rooms
    items.push(formatCount(1, 'living room'));
    items.push(formatCount(1, 'master room'));
    const secondaryCount = Math.max(0, state.ldk - 1);
    if(secondaryCount > 0){
      items.push(formatCount(secondaryCount, 'second room'));
    }

    // Bathrooms derived from toilet count (base 1 + selected)
    const bathroomCount = 1 + state.toiletCount;
    items.push(formatCount(bathroomCount, 'bathroom'));

    // Additional checkbox-based rooms
    Object.entries(state.rooms).forEach(([roomName, isSelected]) => {
      if(!isSelected) return;
      const label = ROOM_LABELS[roomName];
      if(label){
        items.push(formatCount(1, label));
      }
    });

    return items.join(', ');
  }

  async function saveRequest(text, totalArea){
    const response = await fetch('/api/save-text', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text, total_area_m2: totalArea })
    });

    if(!response.ok){
      const errorData = await response.json().catch(() => ({}));
      const message = errorData.error || `HTTP ${response.status}`;
      throw new Error(message);
    }

    return response.json();
  }

  function refreshImages(heroUrl, galleryUrl, version, heroVersion, galleryVersion){
    // Use individual versions if available, otherwise use common version or timestamp
    const heroCacheBuster = heroVersion ? `?v=${heroVersion}` : (version ? `?v=${version}` : `?t=${Date.now()}`);
    const galleryCacheBuster = galleryVersion ? `?v=${galleryVersion}` : (version ? `?v=${version}` : `?t=${Date.now()}`);
    
    const heroImg = document.querySelector('#proposal1 img');
    if(heroImg){
      heroImg.src = `${heroUrl}${heroCacheBuster}`;
    }

    const galleryImg = document.querySelector('#galleryBar img');
    if(galleryImg){
      galleryImg.src = `${galleryUrl}${galleryCacheBuster}`;
    }
  }

  function startJobPolling(jobId){
    activeJobId = jobId;
    clearPollTimer();
    const startTime = Date.now();

    const poll = async () => {
      if(activeJobId !== jobId) return;

      try{
        const response = await fetch(`/api/job-status/${jobId}?t=${Date.now()}`, { cache: 'no-store' });
        if(!response.ok){
          throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();

        if(data.status === 'pending'){
          if(Date.now() - startTime > POLL_TIMEOUT_MS){
            throw new Error('Timed out while waiting for the result.');
          }
          setStatus('Processing...', 'info');
          updateLoadingMessage('Generating floorplans...');
          pollTimer = setTimeout(poll, POLL_INTERVAL_MS);
          return;
        }

        if(data.status === 'failed'){
          throw new Error(data.error || 'Job processing failed.');
        }

        if(data.status === 'completed'){
          refreshImages(data.hero_image_url, data.gallery_image_url, data.version, data.hero_version, data.gallery_version);
          setStatus('Done! Images have been refreshed.', 'success');
          hideLoadingOverlay();
          generateBtn.disabled = false;
          return;
        }

        throw new Error('Unknown job status.');
      }catch(err){
        console.error(err);
        setStatus(`Error: ${err.message}`, 'error');
        hideLoadingOverlay();
        generateBtn.disabled = false;
      }
    };

    poll();
  }

  if(unitEl){
    unitEl.addEventListener('click', event => {
      const btn = event.target.closest('.btn-toggle');
      if(!btn) return;
      const unit = btn.dataset.unit;
      if(!unit) return;
      state.unit = unit;
      syncUnitButtons();
    });
  }

  if(floorEl){
    floorEl.addEventListener('click', event => {
      const btn = event.target.closest('.btn-toggle');
      if(!btn || btn.classList.contains('locked')) return;
      const floor = Number(btn.dataset.floor || state.floor);
      state.floor = floor;
      syncFloorButtons();
    });
  }

  if(totalAreaInput){
    totalAreaInput.addEventListener('input', () => {
      const parsed = Number(totalAreaInput.value);
      state.totalArea = Number.isFinite(parsed) ? Math.max(0, Math.floor(parsed)) : 0;
    });
  }

  roomInputs.forEach(input => {
    input.addEventListener('change', () => {
      const roomName = getRoomName(input);
      state.rooms[roomName] = input.checked;
    });
  });

  if(ldkSelect){
    ldkSelect.addEventListener('change', () => {
      const level = Number(ldkSelect.value);
      if(LDK_LEVELS.includes(level)){
        state.ldk = level;
      }
    });
  }

  if(toiletSelect){
    toiletSelect.addEventListener('change', () => {
      const count = Number(toiletSelect.value);
      if(TOILET_OPTIONS.includes(count)){
        state.toiletCount = count;
      }
    });
  }

  initialiseRooms();
  syncAll();

  if(generateBtn){
    generateBtn.addEventListener('click', async () => {
      const description = buildRoomDescription();
      const totalArea = state.totalArea || 100;

      try{
        generateBtn.disabled = true;
        setStatus('Submitting request...', 'info');
        showLoadingOverlay('Submitting request...');
        const data = await saveRequest(description, totalArea);
        setStatus('Request submitted. Waiting for the result...', 'info');
        updateLoadingMessage('Waiting for the prediction...');
        startJobPolling(data.job_id);
      }catch(err){
        console.error(err);
        setStatus('Unable to submit request: ' + err.message, 'error');
        hideLoadingOverlay();
        generateBtn.disabled = false;
      }
    });
  }
})();

// Fade in proposals
window.addEventListener('load', () => {
  document.querySelectorAll('.proposal').forEach(el => el.classList.add('visible'));
});

// Gallery navigation
const galleryImages = Array.from(document.querySelectorAll('#galleryBar img'));
if(galleryImages.length){
  let currentIndex = 0;

  function updateGallery(){
    galleryImages.forEach((img, idx) => {
      img.classList.toggle('active', idx === currentIndex);
    });
  }

  const nextBtn = document.getElementById('next');
  const prevBtn = document.getElementById('prev');

  if(nextBtn){
    nextBtn.addEventListener('click', () => {
      currentIndex = (currentIndex + 1) % galleryImages.length;
      updateGallery();
    });
  }

  if(prevBtn){
    prevBtn.addEventListener('click', () => {
      currentIndex = (currentIndex - 1 + galleryImages.length) % galleryImages.length;
      updateGallery();
    });
  }

  galleryImages.forEach((img, idx) => {
    img.addEventListener('click', () => {
      currentIndex = idx;
      updateGallery();

      const proposalId = `proposal${idx + 1}`;
      const proposalElem = document.getElementById(proposalId);
      if(proposalElem){
        proposalElem.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  updateGallery();
}

// Close gallery bar
const closeGalleryBtn = document.getElementById('closeGallery');
if(closeGalleryBtn){
  closeGalleryBtn.addEventListener('click', () => {
    const galleryBar = document.getElementById('galleryBar');
    if(galleryBar){
      galleryBar.style.display = 'none';
    }
  });
}

// Land Setting Modal Functions
let headerHidden = false;
let galleryBarHidden = false;

function openLandModal() {
  document.getElementById('landModal').style.display = 'flex';
  updateLandGrid();

  // Hide header and gallery-bar
  const header = document.querySelector('.header');
  const galleryBar = document.getElementById('galleryBar');
  if (header) {
    header.style.display = 'none';
    headerHidden = true;
  }
  if (galleryBar) {
    galleryBar.style.display = 'none';
    galleryBarHidden = true;
  }
}

function closeLandModal() {
  document.getElementById('landModal').style.display = 'none';

  // Restore header and gallery-bar
  const header = document.querySelector('.header');
  const galleryBar = document.getElementById('galleryBar');
  if (headerHidden && header) {
    header.style.display = '';
  }
  if (galleryBarHidden && galleryBar) {
    galleryBar.style.display = '';
  }
  headerHidden = false;
  galleryBarHidden = false;
}

// Add click event to modal background to close when clicking outside
document.addEventListener('DOMContentLoaded', function() {
  const modal = document.getElementById('landModal');
  if (modal) {
    modal.addEventListener('click', function(event) {
      if (event.target === modal) {
        closeLandModal();
      }
    });
  }
});

function updateArea() {
  const vertical = parseInt(document.getElementById('vertical').value) || 10;
  const yoko = parseInt(document.getElementById('yoko').value) || 10;
  const area = vertical * yoko;
  document.getElementById('area').textContent = area;
  updateLandGrid();
}

function updateLandGrid() {
  const vertical = parseInt(document.getElementById('vertical').value) || 10;
  const yoko = parseInt(document.getElementById('yoko').value) || 10;
  const grid = document.getElementById('landGrid');
  grid.innerHTML = '';
  grid.style.gridTemplateColumns = `repeat(${yoko}, 25px)`;
  grid.style.gridTemplateRows = `repeat(${vertical}, 25px)`;

  for (let i = 0; i < vertical * yoko; i++) {
    const cell = document.createElement('div');
    cell.addEventListener('click', () => {
      cell.classList.toggle('off-site');
      if (cell.classList.contains('off-site')) {
        cell.style.backgroundColor = '#fff';
      } else {
        cell.style.backgroundColor = '#fef3c7';
      }
    });
    grid.appendChild(cell);
  }
}

function saveLandSettings() {
  const vertical = parseInt(document.getElementById('vertical').value);
  const yoko = parseInt(document.getElementById('yoko').value);
  const grid = document.getElementById('landGrid');
  const cells = Array.from(grid.children);

  // Build 2D grid array
  const gridData = [];
  for (let i = 0; i < vertical; i++) {
    const row = [];
    for (let j = 0; j < yoko; j++) {
      const cell = cells[i * yoko + j];
      row.push(cell.classList.contains('off-site') ? 0 : 1);
    }
    gridData.push(row);
  }

  const data = {
    length: vertical,
    width: yoko,
    grid: gridData
  };

  fetch('/api/save-land-settings', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  })
    .then(response => {
      if (!response.ok) {
        return response.json().then(err => {
          const message = err && err.error ? err.error : `HTTP ${response.status}`;
          throw new Error(message);
        }).catch(() => {
          throw new Error(`HTTP ${response.status}`);
        });
      }
      return response.json();
    })
    .then(() => {
      closeLandModal();
    })
    .catch(err => {
      console.error('Failed to save land settings:', err);
      alert(`Unable to save land settings: ${err.message}`);
    });
}
