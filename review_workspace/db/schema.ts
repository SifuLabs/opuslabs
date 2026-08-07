import { sql } from "drizzle-orm";
import { index, real, sqliteTable, text } from "drizzle-orm/sqlite-core";

export const projects = sqliteTable("projects", {
  id: text("id").primaryKey(),
  name: text("name").notNull(),
  status: text("status").notNull().default("review"),
  createdAt: text("created_at").notNull().default(sql`CURRENT_TIMESTAMP`),
});

export const clips = sqliteTable(
  "clips",
  {
    id: text("id").primaryKey(),
    projectId: text("project_id")
      .notNull()
      .references(() => projects.id, { onDelete: "cascade" }),
    title: text("title").notNull(),
    caption: text("caption").notNull().default(""),
    startTime: real("start_time").notNull().default(0),
    endTime: real("end_time").notNull().default(0),
    score: real("score").notNull().default(0),
    status: text("status").notNull().default("pending"),
    mediaKey: text("media_key"),
    mediaType: text("media_type"),
    createdAt: text("created_at").notNull().default(sql`CURRENT_TIMESTAMP`),
    updatedAt: text("updated_at").notNull().default(sql`CURRENT_TIMESTAMP`),
  },
  (table) => [
    index("idx_clips_project_id").on(table.projectId),
    index("idx_clips_status").on(table.status),
  ],
);

export type Clip = typeof clips.$inferSelect;
