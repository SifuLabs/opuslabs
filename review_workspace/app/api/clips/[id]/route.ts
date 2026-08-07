import { updateClip } from "../../../../db/repository";

const STATUSES = new Set(["pending", "approved", "rejected"]);

export async function PATCH(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await context.params;
    const payload = (await request.json()) as Record<string, unknown>;
    const update: Record<string, string | number> = {};

    if (typeof payload.title === "string" && payload.title.trim()) {
      update.title = payload.title.trim();
    }
    if (typeof payload.caption === "string") update.caption = payload.caption.trim();
    if (typeof payload.startTime === "number" && payload.startTime >= 0) {
      update.startTime = payload.startTime;
    }
    if (typeof payload.endTime === "number" && payload.endTime >= 0) {
      update.endTime = payload.endTime;
    }
    if (typeof payload.status === "string" && STATUSES.has(payload.status)) {
      update.status = payload.status;
    }

    const clip = await updateClip(id, update);
    if (!clip) return Response.json({ error: "Clip not found" }, { status: 404 });
    return Response.json({ clip });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to update clip";
    return Response.json({ error: message }, { status: 500 });
  }
}
