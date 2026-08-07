import { createClip, listClips } from "../../../db/repository";
import { putMedia } from "../../../db/media";

function mediaUrl(key: string | null) {
  return key
    ? `/api/media/${key.split("/").map(encodeURIComponent).join("/")}`
    : null;
}

export async function GET() {
  try {
    const clips = await listClips();
    return Response.json({
      clips: clips.map((clip) => ({ ...clip, mediaUrl: mediaUrl(clip.mediaKey) })),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to load clips";
    return Response.json({ error: message }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const form = await request.formData();
    const file = form.get("file");
    if (!(file instanceof File) || file.size === 0) {
      return Response.json({ error: "A video file is required" }, { status: 400 });
    }

    const uploadId = crypto.randomUUID();
    const projectId = String(form.get("projectId") || crypto.randomUUID());
    const projectName = String(form.get("projectName") || "New review batch").trim();
    const mediaKey = await putMedia(file, uploadId);
    const endTime = Number(form.get("endTime") || 0);
    const clip = await createClip({
      projectId,
      projectName,
      title: String(form.get("title") || file.name.replace(/\.[^.]+$/, "")),
      caption: String(form.get("caption") || ""),
      startTime: Number(form.get("startTime") || 0),
      endTime: Number.isFinite(endTime) ? endTime : 0,
      score: Math.max(0, Math.min(10, Number(form.get("score") || 0))),
      mediaKey,
      mediaType: file.type || "video/mp4",
    });

    return Response.json(
      { clip: { ...clip, mediaUrl: mediaUrl(clip.mediaKey) } },
      { status: 201 },
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to upload clip";
    return Response.json({ error: message }, { status: 500 });
  }
}
