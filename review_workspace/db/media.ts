import { env } from "cloudflare:workers";

export async function putMedia(file: File, clipId: string) {
  const safeName = file.name.replace(/[^a-zA-Z0-9._-]/g, "_");
  const key = `clips/${clipId}/${safeName}`;
  await env.MEDIA.put(key, file.stream(), {
    httpMetadata: { contentType: file.type || "video/mp4" },
  });
  return key;
}

export async function getMedia(key: string) {
  return env.MEDIA.get(key);
}
