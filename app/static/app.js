const STORAGE_KEY = "meeting-analyzer:meetings:v1";
const MAX_FILE_SIZE = 250 * 1024 * 1024;
const ACTIVE_STATUSES = new Set(["fetching", "transcribing", "analyzing", "processing"]);
const POLL_INTERVAL_MS = 1200;

const elements = {
  dropzone: document.querySelector("#dropzone"),
  fileInput: document.querySelector("#file-input"),
  cloudForm: document.querySelector("#cloud-form"),
  cloudUrl: document.querySelector("#cloud-url"),
  cloudSubmit: document.querySelector("#cloud-submit"),
  meetingList: document.querySelector("#meeting-list"),
  emptyState: document.querySelector("#empty-state"),
  count: document.querySelector("#meeting-count"),
  template: document.querySelector("#meeting-template"),
  toast: document.querySelector("#toast"),
};

let meetings = readMeetings();
let isStarting = false;
const pollingTasks = new Set();

function readMeetings() {
  try {
    const data = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    if (!Array.isArray(data)) return [];
    return data.map((meeting) => {
      const migrated = {
        ...meeting,
        source: meeting.source || "manual",
        taskId: meeting.taskId || null,
        normalizedUrl: meeting.normalizedUrl || null,
      };
      if (migrated.status === "processing") {
        if (migrated.taskId) migrated.status = "transcribing";
        else {
          migrated.status = "error";
          migrated.error = "Обработка была прервана. Загрузите файл повторно.";
        }
      }
      return migrated;
    });
  } catch {
    return [];
  }
}

function persistMeetings() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(meetings));
  } catch {
    showToast("Не удалось сохранить результат: хранилище браузера переполнено.");
  }
}

function isActiveStatus(status) {
  return ACTIVE_STATUSES.has(status);
}

function hasActiveMeeting() {
  return meetings.some((meeting) => isActiveStatus(meeting.status));
}

function updateControls() {
  const disabled = isStarting || hasActiveMeeting();
  elements.dropzone.classList.toggle("is-disabled", disabled);
  elements.dropzone.setAttribute("aria-disabled", String(disabled));
  elements.fileInput.disabled = disabled;
  elements.cloudUrl.disabled = disabled;
  elements.cloudSubmit.disabled = disabled;
}

function pluralizeMeetings(count) {
  const mod10 = count % 10;
  const mod100 = count % 100;
  if (mod10 === 1 && mod100 !== 11) return `${count} встреча`;
  if ([2, 3, 4].includes(mod10) && ![12, 13, 14].includes(mod100)) return `${count} встречи`;
  return `${count} встреч`;
}

function formatDate(isoDate) {
  return new Intl.DateTimeFormat("ru-RU", {
    day: "numeric",
    month: "long",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(isoDate));
}

function statusLabel(status) {
  return {
    fetching: "Получение файла",
    transcribing: "Распознавание речи",
    analyzing: "AI-анализ",
    ready: "Готово",
    error: "Ошибка",
  }[status] || "Ошибка";
}

function sourceLabel(source) {
  return source === "cloud" ? "Из облака" : "Загружено вручную";
}

function createList(items) {
  if (!items?.length) {
    const empty = document.createElement("p");
    empty.className = "none";
    empty.textContent = "Не зафиксировано";
    return empty;
  }

  const list = document.createElement("ul");
  items.forEach((item) => {
    const listItem = document.createElement("li");
    listItem.textContent = item;
    list.append(listItem);
  });
  return list;
}

function renderProcessing(container, status) {
  const copy = {
    fetching: ["Получаем видеозапись", "Скачиваем файл и проверяем формат. Анализ начнётся автоматически."],
    transcribing: ["Распознаём речь", "Извлекаем аудио и преобразуем разговор в текст."],
    analyzing: ["Составляем конспект", "Выделяем решения, задачи и открытые вопросы."],
  }[status] || ["Обрабатываем запись", "Это может занять несколько минут."];

  container.innerHTML = `
    <div class="processing-view">
      <span class="processing-spinner" aria-hidden="true"></span>
      <div>
        <strong class="body-title"></strong>
        <p class="body-copy"></p>
      </div>
    </div>`;
  container.querySelector(".body-title").textContent = copy[0];
  container.querySelector(".body-copy").textContent = copy[1];
}

function renderError(container, meeting) {
  const view = document.createElement("div");
  view.className = "error-view";
  view.innerHTML = `
    <span class="error-dot" aria-hidden="true">!</span>
    <div>
      <strong class="body-title">Не удалось обработать запись</strong>
      <p class="body-copy"></p>
      <button class="retry-button" type="button"></button>
    </div>`;
  view.querySelector(".body-copy").textContent = meeting.error || "Произошла неизвестная ошибка.";
  const retryButton = view.querySelector(".retry-button");
  retryButton.textContent = meeting.source === "cloud" ? "Загрузить с компьютера" : "Загрузить файл повторно";
  retryButton.addEventListener("click", (event) => {
    event.stopPropagation();
    if (!hasActiveMeeting()) elements.fileInput.click();
  });
  container.replaceChildren(view);
}

function makeSummaryBlock(title, items) {
  const section = document.createElement("section");
  section.className = "summary-block";
  const heading = document.createElement("h4");
  heading.textContent = title;
  section.append(heading, createList(items));
  return section;
}

function renderResult(container, meeting) {
  const result = document.createElement("div");
  result.className = "result-view";
  const brief = document.createElement("p");
  brief.className = "summary-brief";
  brief.textContent = meeting.summary?.brief || "Краткое содержание не сформировано.";

  const grid = document.createElement("div");
  grid.className = "summary-grid";
  grid.append(
    makeSummaryBlock("Решения", meeting.summary?.decisions),
    makeSummaryBlock("Задачи", meeting.summary?.tasks),
    makeSummaryBlock("Открытые вопросы", meeting.summary?.open_questions),
  );

  const details = document.createElement("details");
  details.className = "transcript-details";
  const summary = document.createElement("summary");
  summary.textContent = "Показать расшифровку";
  const transcript = document.createElement("p");
  transcript.className = "transcript-text";
  transcript.textContent = meeting.transcript || "Расшифровка отсутствует.";
  details.append(summary, transcript);
  result.append(brief, grid, details);
  container.replaceChildren(result);
}

function renderMeetings(expandedId = null) {
  elements.meetingList.replaceChildren();
  elements.emptyState.hidden = meetings.length > 0;
  elements.count.textContent = pluralizeMeetings(meetings.length);

  meetings.forEach((meeting) => {
    const fragment = elements.template.content.cloneNode(true);
    const card = fragment.querySelector(".meeting-card");
    const header = fragment.querySelector(".meeting-header");
    const body = fragment.querySelector(".meeting-body");
    const badge = fragment.querySelector(".status-badge");
    card.dataset.meetingId = meeting.id;

    fragment.querySelector(".meeting-name").textContent = meeting.filename;
    fragment.querySelector(".meeting-source").textContent = sourceLabel(meeting.source);
    fragment.querySelector(".meeting-date").textContent = `Добавлено ${formatDate(meeting.createdAt)}`;
    badge.textContent = statusLabel(meeting.status);
    badge.classList.add(`status-${meeting.status}`);

    if (isActiveStatus(meeting.status)) renderProcessing(body, meeting.status);
    if (meeting.status === "error") renderError(body, meeting);
    if (meeting.status === "ready") renderResult(body, meeting);

    const shouldExpand = meeting.id === expandedId || isActiveStatus(meeting.status);
    header.setAttribute("aria-expanded", String(shouldExpand));
    body.hidden = !shouldExpand;
    header.addEventListener("click", () => {
      const isExpanded = header.getAttribute("aria-expanded") === "true";
      header.setAttribute("aria-expanded", String(!isExpanded));
      body.hidden = isExpanded;
    });
    elements.meetingList.append(fragment);
  });
  updateControls();
}

function openMeeting(meetingId) {
  renderMeetings(meetingId);
  requestAnimationFrame(() => {
    const card = elements.meetingList.querySelector(`[data-meeting-id="${meetingId}"]`);
    card?.scrollIntoView({ behavior: "smooth", block: "center" });
  });
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("is-visible");
  window.clearTimeout(showToast.timeoutId);
  showToast.timeoutId = window.setTimeout(() => elements.toast.classList.remove("is-visible"), 4600);
}

function getErrorMessage(payload, fallback) {
  return payload && typeof payload.detail === "string" ? payload.detail : fallback;
}

function validateFile(file) {
  const extension = file.name.split(".").pop()?.toLowerCase();
  if (!extension || !["webm", "mp4"].includes(extension)) return "Выберите файл WebM или MP4.";
  if (file.size === 0) return "Файл пуст. Выберите другую видеозапись.";
  if (file.size > MAX_FILE_SIZE) return "Файл слишком большой. Максимальный размер — 250 МБ.";
  return null;
}

function normalizeGoogleDriveUrl(rawUrl) {
  let url;
  try {
    url = new URL(rawUrl.trim());
  } catch {
    throw new Error("Введите корректную ссылку на Google Drive.");
  }
  if (url.protocol !== "https:" || !["drive.google.com", "drive.usercontent.google.com"].includes(url.hostname)) {
    throw new Error("Поддерживаются только HTTPS-ссылки на файлы Google Drive.");
  }
  const pathMatch = url.pathname.match(/\/file\/d\/([A-Za-z0-9_-]+)/);
  const fileId = pathMatch?.[1] || url.searchParams.get("id") || "";
  if (!/^[A-Za-z0-9_-]{10,}$/.test(fileId)) {
    throw new Error("Не удалось определить файл. Используйте ссылку Google Drive вида /file/d/…/view.");
  }
  return `https://drive.google.com/file/d/${fileId}`;
}

function createLocalMeeting({ filename, source, normalizedUrl = null }) {
  return {
    id: crypto.randomUUID(),
    taskId: null,
    filename,
    source,
    normalizedUrl,
    createdAt: new Date().toISOString(),
    status: "fetching",
    transcript: "",
    summary: null,
    error: "",
  };
}

function applyTaskPayload(meeting, payload) {
  meeting.taskId = payload.id || meeting.taskId;
  meeting.filename = payload.filename || meeting.filename;
  meeting.source = payload.source || meeting.source;
  meeting.normalizedUrl = payload.normalized_url || meeting.normalizedUrl;
  meeting.status = payload.status || meeting.status;
  meeting.transcript = payload.transcript || meeting.transcript;
  meeting.summary = payload.summary || meeting.summary;
  meeting.error = payload.error || "";
}

const delay = (milliseconds) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));

async function pollTask(meeting) {
  if (!meeting.taskId || pollingTasks.has(meeting.taskId)) return;
  pollingTasks.add(meeting.taskId);
  try {
    while (isActiveStatus(meeting.status)) {
      const response = await fetch(`/api/tasks/${encodeURIComponent(meeting.taskId)}`);
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(getErrorMessage(payload, "Не удалось получить состояние обработки."));
      applyTaskPayload(meeting, payload);
      persistMeetings();
      renderMeetings(meeting.id);
      if (isActiveStatus(meeting.status)) await delay(POLL_INTERVAL_MS);
    }
    if (meeting.status === "ready") showToast("Встреча обработана — конспект готов.");
  } catch (error) {
    meeting.status = "error";
    meeting.error = error instanceof TypeError
      ? "Нет связи с сервером. Перезапустите приложение и добавьте запись повторно."
      : error.message;
    persistMeetings();
    renderMeetings(meeting.id);
  } finally {
    pollingTasks.delete(meeting.taskId);
    updateControls();
  }
}

async function startManualUpload(file) {
  const validationError = validateFile(file);
  if (validationError) return showToast(validationError);
  if (hasActiveMeeting() || isStarting) return showToast("Дождитесь завершения текущей обработки.");

  isStarting = true;
  const meeting = createLocalMeeting({ filename: file.name, source: "manual" });
  meetings.unshift(meeting);
  persistMeetings();
  renderMeetings(meeting.id);
  document.querySelector(".meetings-section").scrollIntoView({ behavior: "smooth", block: "start" });

  const formData = new FormData();
  formData.append("file", file, file.name);
  try {
    const response = await fetch("/api/tasks/upload", { method: "POST", body: formData });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(getErrorMessage(payload, "Сервер не смог принять запись."));
    applyTaskPayload(meeting, payload);
    persistMeetings();
    renderMeetings(meeting.id);
    pollTask(meeting);
  } catch (error) {
    meeting.status = "error";
    meeting.error = error instanceof TypeError ? "Нет связи с сервером." : error.message;
    persistMeetings();
    renderMeetings(meeting.id);
  } finally {
    isStarting = false;
    elements.fileInput.value = "";
    updateControls();
  }
}

async function startCloudImport(rawUrl) {
  let normalizedUrl;
  try {
    normalizedUrl = normalizeGoogleDriveUrl(rawUrl);
  } catch (error) {
    return showToast(error.message);
  }

  const existing = meetings.find(
    (meeting) => meeting.normalizedUrl === normalizedUrl && meeting.status !== "error",
  );
  if (existing) {
    showToast(existing.status === "ready"
      ? "Эта запись уже обработана — открываем готовый результат."
      : "Эта запись уже находится в обработке.");
    openMeeting(existing.id);
    return;
  }
  if (hasActiveMeeting() || isStarting) return showToast("Дождитесь завершения текущей обработки.");

  isStarting = true;
  const meeting = createLocalMeeting({
    filename: "Видео из Google Drive",
    source: "cloud",
    normalizedUrl,
  });
  meetings.unshift(meeting);
  persistMeetings();
  renderMeetings(meeting.id);
  document.querySelector(".meetings-section").scrollIntoView({ behavior: "smooth", block: "start" });

  try {
    const response = await fetch("/api/tasks/cloud", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: rawUrl.trim() }),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(getErrorMessage(payload, "Не удалось добавить файл из облака."));
    applyTaskPayload(meeting, payload);
    elements.cloudUrl.value = "";
    persistMeetings();
    renderMeetings(meeting.id);
    if (isActiveStatus(meeting.status)) pollTask(meeting);
    else if (meeting.status === "ready") showToast("Открыт уже готовый результат этой записи.");
  } catch (error) {
    meeting.status = "error";
    meeting.error = error instanceof TypeError ? "Нет связи с сервером." : error.message;
    persistMeetings();
    renderMeetings(meeting.id);
  } finally {
    isStarting = false;
    updateControls();
  }
}

elements.dropzone.addEventListener("click", () => {
  if (!hasActiveMeeting() && !isStarting) elements.fileInput.click();
});
elements.dropzone.addEventListener("keydown", (event) => {
  if ((event.key === "Enter" || event.key === " ") && !hasActiveMeeting() && !isStarting) {
    event.preventDefault();
    elements.fileInput.click();
  }
});
elements.fileInput.addEventListener("change", () => {
  const [file] = elements.fileInput.files;
  if (file) startManualUpload(file);
});
elements.cloudForm.addEventListener("submit", (event) => {
  event.preventDefault();
  startCloudImport(elements.cloudUrl.value);
});

["dragenter", "dragover"].forEach((eventName) => {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    if (!hasActiveMeeting() && !isStarting) elements.dropzone.classList.add("is-dragging");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.remove("is-dragging");
  });
});
elements.dropzone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file) startManualUpload(file);
});

persistMeetings();
renderMeetings();
meetings.filter((meeting) => isActiveStatus(meeting.status)).forEach((meeting) => {
  if (meeting.taskId) pollTask(meeting);
  else {
    meeting.status = "error";
    meeting.error = "Не удалось восстановить задачу. Добавьте запись повторно.";
  }
});
persistMeetings();
renderMeetings();
