"use client";

import { ChangeEvent, DragEvent, useEffect, useMemo, useState } from "react";
import { ClipCard } from "./ClipCard";
import type { ClipUpdate, ReviewClip, ReviewStatus } from "./review-types";

type Filter = "all" | ReviewStatus;

export function ReviewWorkspace() {
  const [clips, setClips] = useState<ReviewClip[]>([]);
  const [filter, setFilter] = useState<Filter>("all");
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");

  async function loadClips() {
    try {
      const response = await fetch("/api/clips", { cache: "no-store" });
      const payload = (await response.json()) as { clips?: ReviewClip[]; error?: string };
      if (!response.ok) throw new Error(payload.error || "Could not load the review queue");
      setClips(payload.clips || []);
      setError("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load clips");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadClips();
  }, []);

  const totals = useMemo(() => ({
    all: clips.length,
    pending: clips.filter((clip) => clip.status === "pending").length,
    approved: clips.filter((clip) => clip.status === "approved").length,
    rejected: clips.filter((clip) => clip.status === "rejected").length,
  }), [clips]);

  const visibleClips = filter === "all"
    ? clips
    : clips.filter((clip) => clip.status === filter);

  async function updateClip(id: string, update: ClipUpdate) {
    const response = await fetch(`/api/clips/${id}`, {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(update),
    });
    const payload = (await response.json()) as { clip?: ReviewClip; error?: string };
    if (!response.ok || !payload.clip) throw new Error(payload.error || "Update failed");
    setClips((current) => current.map((clip) => (
      clip.id === id ? { ...clip, ...payload.clip, mediaUrl: clip.mediaUrl } : clip
    )));
  }

  async function uploadFiles(files: FileList | File[]) {
    const videos = Array.from(files).filter((file) => file.type.startsWith("video/"));
    if (!videos.length) {
      setError("Choose one or more video clips to add to review.");
      return;
    }

    setUploading(true);
    setError("");
    const projectId = crypto.randomUUID();
    try {
      for (const file of videos) {
        const form = new FormData();
        form.set("file", file);
        form.set("projectId", projectId);
        form.set("projectName", `Review batch ${new Date().toLocaleDateString()}`);
        form.set("title", file.name.replace(/\.[^.]+$/, ""));
        const response = await fetch("/api/clips", { method: "POST", body: form });
        const payload = (await response.json()) as { error?: string };
        if (!response.ok) throw new Error(payload.error || `Could not upload ${file.name}`);
      }
      await loadClips();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  function onFileInput(event: ChangeEvent<HTMLInputElement>) {
    if (event.target.files) void uploadFiles(event.target.files);
    event.target.value = "";
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    void uploadFiles(event.dataTransfer.files);
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#top" aria-label="OpusLabs review home">
          <span className="brand-mark">O</span>
          <span>Opus<span>Labs</span></span>
        </a>
        <nav aria-label="Workspace navigation">
          <a className="active" href="#queue"><span>01</span> Review queue</a>
          <a href="#approved"><span>02</span> Approved</a>
          <a href="#exports"><span>03</span> Exports</a>
        </nav>
        <div className="sidebar-note">
          <span className="pulse" />
          <p>Review workspace</p>
          <small>Edits save automatically</small>
        </div>
      </aside>

      <section className="workspace" id="top">
        <header className="topbar">
          <div>
            <p className="eyebrow">Video editing copilot</p>
            <h1>Shape the cut before it ships.</h1>
          </div>
          <label className={`upload-button ${uploading ? "busy" : ""}`}>
            <input type="file" accept="video/*" multiple onChange={onFileInput} disabled={uploading} />
            <span>{uploading ? "Uploading..." : "Add clips"}</span>
            <b>+</b>
          </label>
        </header>

        <section className="overview" aria-label="Review overview">
          <div className="overview-copy">
            <span className="section-index">CURRENT BATCH / REVIEW</span>
            <h2>Keep the moments<br />that earn attention.</h2>
            <p>Watch every candidate, tighten the timing, polish the copy, then approve only the clips ready to publish.</p>
          </div>
          <div className="metric-grid">
            <div><strong>{totals.pending}</strong><span>Awaiting review</span></div>
            <div><strong>{totals.approved}</strong><span>Ready to export</span></div>
            <div><strong>{clips.length ? Math.round((totals.approved / clips.length) * 100) : 0}%</strong><span>Approval rate</span></div>
          </div>
        </section>

        <label className="drop-zone" onDragOver={(event) => event.preventDefault()} onDrop={onDrop}>
          <input type="file" accept="video/*" multiple onChange={onFileInput} disabled={uploading} />
          <span className="drop-icon">+</span>
          <span><strong>Drop finished candidates here</strong><small>MP4, MOV or WebM. Add several clips as one review batch.</small></span>
          <b>Browse files</b>
        </label>

        <section className="queue" id="queue">
          <div className="queue-header">
            <div><p className="eyebrow">Editorial pass</p><h2>Clip review queue</h2></div>
            <div className="filters" aria-label="Filter clips">
              {(["all", "pending", "approved", "rejected"] as Filter[]).map((item) => (
                <button key={item} className={filter === item ? "active" : ""} onClick={() => setFilter(item)}>
                  {item === "rejected" ? "Passed" : item[0].toUpperCase() + item.slice(1)}
                  <span>{totals[item]}</span>
                </button>
              ))}
            </div>
          </div>

          {error && <div className="error-banner" role="alert">{error}</div>}
          {loading ? (
            <div className="empty-state"><span className="loader" /><p>Loading your review queue...</p></div>
          ) : visibleClips.length ? (
            <div className="clip-grid">
              {visibleClips.map((clip) => <ClipCard key={clip.id} clip={clip} onUpdate={updateClip} />)}
            </div>
          ) : (
            <div className="empty-state">
              <span className="empty-number">00</span>
              <h3>{filter === "all" ? "Your strongest cuts start here." : `No ${filter} clips yet.`}</h3>
              <p>{filter === "all" ? "Add generated clips to start the editorial review." : "Choose another filter or update a clip decision."}</p>
            </div>
          )}
        </section>
      </section>
    </main>
  );
}
