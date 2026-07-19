/**
 * GET /api/history/categories — category buckets + active-session counts
 *
 * Drives the sidebar's category chips. ``null`` category means the
 * "No category" bucket and is always shown last in the UI even when
 * its count is largest.
 *
 * Response
 *     { categories: [{ category: string | null, count: number }, ...] }
 */
import type { NextRequest } from "next/server";

import { fdUidOf, requireAuth, requireFdUid } from "@/lib/auth";
import { listCategoriesWithCounts } from "@/lib/history-db";

export const runtime = "edge";

export async function GET(req: NextRequest): Promise<Response> {
  const authFail = requireAuth(req);
  if (authFail) return authFail;
  const uidFail = requireFdUid(req);
  if (uidFail) return uidFail;
  const uid = fdUidOf(req) as string;

  try {
    const categories = await listCategoriesWithCounts(uid);
    return Response.json({ categories });
  } catch (e) {
    return Response.json(
      { error: e instanceof Error ? e.message : "listCategories failed" },
      { status: 500 },
    );
  }
}
