import { env } from "cloudflare:workers";
import { desc, eq } from "drizzle-orm";
import { getDb } from ".";
import { clips, projects } from "./schema";

export async function ensureSchema() {
  await env.DB.batch([
    env.DB.prepare(`CREATE TABLE IF NOT EXISTS projects (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      status TEXT NOT NULL DEFAULT 'review',
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )`),
    env.DB.prepare(`CREATE TABLE IF NOT EXISTS clips (
      id TEXT PRIMARY KEY,
      project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
      title TEXT NOT NULL,
      caption TEXT NOT NULL DEFAULT '',
      start_time REAL NOT NULL DEFAULT 0,
      end_time REAL NOT NULL DEFAULT 0,
      score REAL NOT NULL DEFAULT 0,
      status TEXT NOT NULL DEFAULT 'pending',
      media_key TEXT,
      media_type TEXT,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )`),
    env.DB.prepare("CREATE INDEX IF NOT EXISTS idx_clips_project_id ON clips(project_id)"),
    env.DB.prepare("CREATE INDEX IF NOT EXISTS idx_clips_status ON clips(status)"),
  ]);
}

export async function listClips() {
  await ensureSchema();
  return getDb().select().from(clips).orderBy(desc(clips.createdAt));
}

export async function createClip(input: {
  projectId: string;
  projectName: string;
  title: string;
  caption: string;
  startTime: number;
  endTime: number;
  score: number;
  mediaKey: string | null;
  mediaType: string | null;
}) {
  await ensureSchema();
  const db = getDb();
  await db.insert(projects).values({
    id: input.projectId,
    name: input.projectName,
  }).onConflictDoNothing();
  const id = crypto.randomUUID();
  const [clip] = await db.insert(clips).values({ id, ...input }).returning();
  return clip;
}

export async function updateClip(id: string, input: Partial<{
  title: string;
  caption: string;
  startTime: number;
  endTime: number;
  status: string;
}>) {
  await ensureSchema();
  const [clip] = await getDb().update(clips).set({
    ...input,
    updatedAt: new Date().toISOString(),
  }).where(eq(clips.id, id)).returning();
  return clip;
}
