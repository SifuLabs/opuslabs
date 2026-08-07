export type ReviewStatus = "pending" | "approved" | "rejected";

export type ReviewClip = {
  id: string;
  projectId: string;
  title: string;
  caption: string;
  startTime: number;
  endTime: number;
  score: number;
  status: ReviewStatus;
  mediaUrl: string | null;
  createdAt: string;
};

export type ClipUpdate = Partial<
  Pick<ReviewClip, "title" | "caption" | "startTime" | "endTime" | "status">
>;
