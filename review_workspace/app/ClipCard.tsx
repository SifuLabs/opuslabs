"use client";

import { useEffect, useState } from "react";
import type { ClipUpdate, ReviewClip, ReviewStatus } from "./review-types";

type Props = {
  clip: ReviewClip;
  onUpdate: (id: string, update: ClipUpdate) => Promise<void>;
};

const statusLabels: Record<ReviewStatus, string> = {
  pending: "Needs review",
  approved: "Approved",
  rejected: "Passed",
};

function formatTime(seconds: number) {
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.max(0, seconds % 60);
  return `${minutes}:${remainder.toFixed(1).padStart(4, "0")}`;
}

export function ClipCard({ clip, onUpdate }: Props) {
  const [title, setTitle] = useState(clip.title);
  const [caption, setCaption] = useState(clip.caption);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setTitle(clip.title);
    setCaption(clip.caption);
  }, [clip.title, clip.caption]);

  async function save(update: ClipUpdate) {
    setSaving(true);
    try {
      await onUpdate(clip.id, update);
    } finally {
      setSaving(false);
    }
  }

  return (
    <article className={`clip-card status-${clip.status}`}>
      <div className="clip-preview">
        {clip.mediaUrl ? (
          <video controls preload="metadata" src={clip.mediaUrl} aria-label={`Preview ${clip.title}`} />
        ) : (
          <div className="video-placeholder" aria-label="Video preview unavailable">
            <span>Preview processing</span>
          </div>
        )}
        <span className="duration-badge">{formatTime(clip.endTime - clip.startTime)}</span>
        <span className={`status-badge ${clip.status}`}>{statusLabels[clip.status]}</span>
      </div>

      <div className="clip-details">
        <div className="clip-heading-row">
          <input
            className="title-input"
            value={title}
            onChange={(event) => setTitle(event.target.value)}
            onBlur={() => title.trim() && title !== clip.title && save({ title })}
            aria-label="Clip title"
          />
          <div className="score" title="Predicted engagement score">
            <span>{clip.score.toFixed(1)}</span><small>/10</small>
          </div>
        </div>

        <label className="caption-field">
          <span>Post caption</span>
          <textarea
            value={caption}
            onChange={(event) => setCaption(event.target.value)}
            onBlur={() => caption !== clip.caption && save({ caption })}
            placeholder="Add a publish-ready caption..."
          />
        </label>

        <div className="trim-row">
          <label>
            In
            <input
              type="number"
              min="0"
              step="0.1"
              defaultValue={clip.startTime}
              onBlur={(event) => save({ startTime: Number(event.target.value) })}
            />
          </label>
          <span className="trim-line" />
          <label>
            Out
            <input
              type="number"
              min="0"
              step="0.1"
              defaultValue={clip.endTime}
              onBlur={(event) => save({ endTime: Number(event.target.value) })}
            />
          </label>
          <span className="save-state">{saving ? "Saving..." : "Saved"}</span>
        </div>

        <div className="clip-actions">
          <button
            className="reject-button"
            onClick={() => save({ status: clip.status === "rejected" ? "pending" : "rejected" })}
          >
            {clip.status === "rejected" ? "Restore" : "Pass"}
          </button>
          <button
            className="approve-button"
            onClick={() => save({ status: clip.status === "approved" ? "pending" : "approved" })}
          >
            {clip.status === "approved" ? "Undo approval" : "Approve clip"}
          </button>
        </div>
      </div>
    </article>
  );
}
