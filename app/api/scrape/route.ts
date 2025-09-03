// Server-side scraper using fetch + cheerio.
// Extracts profiles via JSON-LD (schema.org Person), Microdata Person, h-card microformats, and heuristics.

import type { NextRequest } from "next/server"
import * as cheerio from "cheerio"

type SocialLink = { type: string; url: string }
type Profile = {
  name?: string
  title?: string
  email?: string
  image?: string
  bio?: string
  socials?: SocialLink[]
  links?: string[]
  source?: string
}

const ALLOWED_PROTOCOLS = new Set(["http:", "https:"])

// Basic SSRF guard: block localhost/private IP literals and localhost hostnames
function isBlockedHost(u: URL) {
  const host = u.hostname.toLowerCase()
  if (host === "localhost" || host.endsWith(".localhost")) return true
  if (host === "0.0.0.0") return true
  if (host === "127.0.0.1" || host.startsWith("127.")) return true
  // Private ranges if provided as literal IP
  const parts = host.split(".").map((p) => Number(p))
  if (parts.length === 4 && parts.every((n) => Number.isFinite(n))) {
    const [a, b] = parts
    if (a === 10) return true
    if (a === 172 && b >= 16 && b <= 31) return true
    if (a === 192 && b === 168) return true
  }
  return false
}

function safeURL(input: string): URL | null {
  try {
    const u = new URL(input)
    if (!ALLOWED_PROTOCOLS.has(u.protocol)) return null
    if (isBlockedHost(u)) return null
    return u
  } catch {
    return null
  }
}

function absUrl(base: URL, href?: string | null): string | undefined {
  if (!href) return undefined
  try {
    return new URL(href, base).toString()
  } catch {
    return undefined
  }
}

function parseEmail(href?: string | null): string | undefined {
  if (!href) return undefined
  if (href.startsWith("mailto:")) {
    return href.slice("mailto:".length).split("?")[0]
  }
  // Fallback: detect visible email text
  const m = href.match(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i)
  return m ? m[0] : undefined
}

function detectSocial(url: string): SocialLink | null {
  const u = url.toLowerCase()
  if (u.includes("github.com/")) return { type: "GitHub", url }
  if (u.includes("linkedin.com/in") || u.includes("linkedin.com/company")) return { type: "LinkedIn", url }
  if (u.includes("twitter.com/") || u.includes("x.com/")) return { type: "Twitter/X", url }
  if (u.includes("instagram.com/")) return { type: "Instagram", url }
  if (u.includes("facebook.com/")) return { type: "Facebook", url }
  if (u.includes("youtube.com/") || u.includes("youtu.be/")) return { type: "YouTube", url }
  if (u.includes("medium.com/")) return { type: "Medium", url }
  if (u.includes("t.me/")) return { type: "Telegram", url }
  if (u.includes("mastodon.social") || u.includes("@")) return null // too ambiguous; skip generic @ links
  return null
}

function cleanText(s?: string | null): string | undefined {
  if (!s) return undefined
  const t = s.replace(/\s+/g, " ").trim()
  return t || undefined
}

function uniq<T>(arr: T[], key: (v: T) => string): T[] {
  const m = new Map<string, T>()
  for (const v of arr) m.set(key(v), v)
  return [...m.values()]
}

function mergeProfiles(list: Profile[]): Profile[] {
  // Merge by (name + email) or shared social url
  const out: Profile[] = []
  for (const p of list) {
    let merged = false
    for (const q of out) {
      const sameNameEmail =
        p.name &&
        q.name &&
        p.name.toLowerCase() === q.name.toLowerCase() &&
        ((p.email && q.email && p.email.toLowerCase() === q.email.toLowerCase()) || (!p.email && !q.email))
      const socialOverlap = new Set((q.socials || []).map((s) => s.url))
      const hasOverlap = (p.socials || []).some((s) => socialOverlap.has(s.url))
      if (sameNameEmail || hasOverlap) {
        q.name ||= p.name
        q.title ||= p.title
        q.email ||= p.email
        q.image ||= p.image
        q.bio ||= p.bio
        q.socials = uniq([...(q.socials || []), ...(p.socials || [])], (s) => s.url)
        q.links = uniq([...(q.links || []), ...(p.links || [])], (l) => l)
        merged = true
        break
      }
    }
    if (!merged) out.push({ ...p, socials: uniq(p.socials || [], (s) => s.url), links: uniq(p.links || [], (l) => l) })
  }
  return out
}

// Extractors

function extractJsonLd($: cheerio.CheerioAPI, base: URL): Profile[] {
  const profiles: Profile[] = []
  $('script[type="application/ld+json"]').each((_, el) => {
    const raw = $(el).contents().text()
    if (!raw) return
    try {
      const json = JSON.parse(raw)
      const items = Array.isArray(json) ? json : [json]
      for (const item of items) {
        const queue: any[] = [item]
        while (queue.length) {
          const cur = queue.shift()
          if (!cur || typeof cur !== "object") continue
          const type = cur["@type"] || cur.type
          const types = Array.isArray(type) ? type : [type]
          if (types && types.some((t: string) => typeof t === "string" && t.toLowerCase().includes("person"))) {
            const p: Profile = {
              name: cleanText(cur.name),
              title: cleanText(cur.jobTitle || cur.jobtitle),
              email: cleanText(cur.email),
              image: absUrl(base, cur.image?.url || cur.image),
              bio: cleanText(cur.description),
              socials: [],
              links: [],
              source: "jsonld",
            }
            const sameAs: string[] = []
            if (Array.isArray(cur.sameAs)) sameAs.push(...cur.sameAs)
            if (cur.url) sameAs.push(cur.url)
            for (const u of sameAs) {
              const full = absUrl(base, u)
              if (!full) continue
              const s = detectSocial(full)
              if (s) p.socials!.push(s)
              else p.links!.push(full)
            }
            profiles.push(p)
          }
          for (const v of Object.values(cur)) if (v && typeof v === "object") queue.push(v)
        }
      }
    } catch {
      // Ignore malformed JSON-LD blocks
    }
  })
  return profiles
}

function extractMicrodata($: cheerio.CheerioAPI, base: URL): Profile[] {
  const profiles: Profile[] = []
  const sel = '[itemscope][itemtype*="schema.org/Person"], [itemtype*="schema.org/person"]'
  $(sel).each((_, el) => {
    const node = $(el)
    const getProp = (prop: string) => {
      const t = node.find(`[itemprop="${prop}"]`).first()
      if (!t.length) return undefined
      if (t.is("meta")) return cleanText(t.attr("content") || t.attr("value"))
      if (t.is("img")) return absUrl(base, t.attr("src"))
      return cleanText(t.text() || t.attr("content") || t.attr("value"))
    }
    const p: Profile = {
      name: cleanText(getProp("name")),
      title: cleanText(getProp("jobTitle")),
      email: cleanText(getProp("email")),
      image: getProp("image"),
      bio: cleanText(getProp("description")),
      socials: [],
      links: [],
      source: "microdata",
    }
    node.find('[itemprop="sameAs"], a[itemprop="url"]').each((_, a) => {
      const href = $(a).attr("href")
      const full = absUrl(base, href)
      if (!full) return
      const s = detectSocial(full)
      if (s) p.socials!.push(s)
      else p.links!.push(full)
    })
    profiles.push(p)
  })
  return profiles
}

function extractHCard($: cheerio.CheerioAPI, base: URL): Profile[] {
  const profiles: Profile[] = []
  $(".h-card").each((_, el) => {
    const n = $(el)
    const p: Profile = {
      name: cleanText(n.find(".p-name, .p-org").first().text()),
      title: cleanText(n.find(".p-job-title, .p-title").first().text()),
      email: parseEmail(n.find(".u-email, a[href^='mailto:']").attr("href") || ""),
      image: absUrl(base, n.find(".u-photo, img").first().attr("src")),
      bio: cleanText(n.find(".p-note, .p-bio").first().text()),
      socials: [],
      links: [],
      source: "h-card",
    }
    n.find("a[rel='me'], a[rel='author'], a").each((_, a) => {
      const full = absUrl(base, $(a).attr("href"))
      if (!full) return
      const s = detectSocial(full)
      if (s) p.socials!.push(s)
      else p.links!.push(full)
    })
    profiles.push(p)
  })
  return profiles
}

function extractHeuristics($: cheerio.CheerioAPI, base: URL): Profile[] {
  const profiles: Profile[] = []
  const candidates = $(
    [
      "[class*='author']",
      "[class*='profile']",
      "[class*='member']",
      "[class*='team']",
      "[class*='person']",
      "[class*='user']",
      "[class*='staff']",
      "[class*='founder']",
      "article.author",
      "section.author",
    ].join(","),
  )

  candidates.each((_, el) => {
    const node = $(el)
    // Look for name and title
    const name = cleanText(node.find("h1, h2, h3, h4, .name, [class*='name']").first().text()) || undefined
    const title =
      cleanText(node.find(".title, .role, [class*='job'], [class*='title'], [class*='role']").first().text()) ||
      undefined
    const email =
      parseEmail(
        node.find("a[href^='mailto:']").attr("href") || node.find("a[href*='@']").attr("href") || node.text(),
      ) || undefined
    const image = absUrl(base, node.find("img").first().attr("src"))
    const bio = cleanText(node.find("p, .bio, [class*='bio'], [class*='about']").first().text()) || undefined

    const socials: SocialLink[] = []
    const links: string[] = []
    node.find("a[href]").each((_, a) => {
      const full = absUrl(base, $(a).attr("href"))
      if (!full) return
      const s = detectSocial(full)
      if (s) socials.push(s)
      else links.push(full)
    })

    if (name || title || email || image || bio || socials.length > 0) {
      profiles.push({
        name,
        title,
        email,
        image,
        bio,
        socials: uniq(socials, (s) => s.url),
        links: uniq(links, (l) => l),
        source: "heuristic",
      })
    }
  })

  // Also detect standalone author blocks
  $("a[rel='author'], [itemprop='author']").each((_, el) => {
    const n = $(el)
    const href = absUrl(base, n.attr("href"))
    const p: Profile = {
      name: cleanText(n.text()),
      links: href ? [href] : [],
      source: "heuristic",
    }
    if (p.name || (p.links && p.links.length)) profiles.push(p)
  })

  return profiles
}

export async function POST(req: NextRequest) {
  try {
    const { url } = (await req.json().catch(() => ({}))) as { url?: string }
    if (!url) {
      return Response.json({ ok: false, error: "Missing 'url' in request body" }, { status: 400 })
    }
    const target = safeURL(url)
    if (!target) {
      return Response.json({ ok: false, error: "Invalid or disallowed URL" }, { status: 400 })
    }

    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 12000)

    const res = await fetch(target.toString(), {
      method: "GET",
      redirect: "follow",
      signal: controller.signal,
      headers: {
        "User-Agent": "ProfileExtractorBot/1.0 (+https://vercel.com/) Mozilla/5.0 (compatible; MesoAssignmentBot/1.0)",
        Accept: "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
      },
    }).catch((e) => {
      if ((e as any).name === "AbortError") throw new Error("Fetch timed out")
      throw e
    })
    clearTimeout(timeout)

    if (!res.ok) {
      return Response.json({ ok: false, error: `Failed to fetch: ${res.status}` }, { status: 502 })
    }

    const contentType = res.headers.get("content-type") || ""
    if (!contentType.includes("text/html")) {
      return Response.json({ ok: false, error: "URL did not return HTML content" }, { status: 400 })
    }

    // Limit body size (first ~1.2MB) to avoid huge pages
    const raw = await res.text()
    const html = raw.length > 1_200_000 ? raw.slice(0, 1_200_000) : raw

    const $ = cheerio.load(html)

    const collected: Profile[] = [
      ...extractJsonLd($, target),
      ...extractMicrodata($, target),
      ...extractHCard($, target),
      ...extractHeuristics($, target),
    ]

    // Clean up results
    const cleaned = mergeProfiles(
      collected
        .map((p) => ({
          ...p,
          name: cleanText(p.name),
          title: cleanText(p.title),
          email: p.email?.toLowerCase(),
          bio: cleanText(p.bio),
          image: p.image,
          socials: uniq(p.socials || [], (s) => s.url),
          links: uniq(p.links || [], (l) => l),
        }))
        .filter((p) => p.name || p.email || (p.socials && p.socials.length) || p.image || p.bio),
    )

    return Response.json({ ok: true, url: target.toString(), profiles: cleaned })
  } catch (e: any) {
    const msg = typeof e?.message === "string" ? e.message : "Unknown error"
    return Response.json({ ok: false, error: msg }, { status: 500 })
  }
}
