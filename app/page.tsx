"use client"

import type React from "react"

import { useState } from "react"
import useSWR from "swr"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import { ProfileCard } from "@/components/profile-card"

// Types mirrored from API
type SocialLink = { type: string; url: string }
export type Profile = {
  name?: string
  title?: string
  email?: string
  image?: string
  bio?: string
  socials?: SocialLink[]
  links?: string[]
  source?: string
}

type ApiResponse = { ok: true; url: string; profiles: Profile[] } | { ok: false; error: string }

const postFetcher = async (key: string): Promise<ApiResponse> => {
  // key looks like /swr/scrape?u=encodedUrl
  const u = new URL(key, typeof window !== "undefined" ? window.location.href : "http://localhost")
  const target = u.searchParams.get("u")
  const res = await fetch("/api/scrape", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url: target }),
  })
  return res.json()
}

export default function HomePage() {
  const [url, setUrl] = useState("")
  const [submitted, setSubmitted] = useState<string | null>(null)

  const { data, error, isValidating, mutate } = useSWR<ApiResponse>(
    submitted ? `/swr/scrape?u=${encodeURIComponent(submitted)}` : null,
    postFetcher,
  )

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!url) return
    setSubmitted(url)
    mutate()
  }

  const profiles = (data && "ok" in data && data.ok ? data.profiles : []) as Profile[]

  return (
    <main className="min-h-dvh bg-background text-foreground">
      <section className="container mx-auto max-w-3xl px-4 py-10">
        <header className="mb-8">
          <h1 className="text-3xl font-semibold text-balance">User Profile Extractor</h1>
          <p className="text-muted-foreground mt-2 text-pretty">
            Enter any website URL and we&apos;ll try to detect and extract user profile information (name, title, email,
            image, bio, social links).
          </p>
        </header>

        <Card>
          <CardHeader>
            <CardTitle>Scan a website</CardTitle>
          </CardHeader>
          <CardContent>
            <form className="flex flex-col gap-4" onSubmit={onSubmit} aria-label="URL scrape form">
              <div className="flex flex-col gap-2">
                <Label htmlFor="url">Website URL</Label>
                <div className="flex items-center gap-2">
                  <Input
                    id="url"
                    type="url"
                    placeholder="https://example.com/team"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    required
                    inputMode="url"
                    aria-describedby="url-hint"
                  />
                  <Button type="submit" disabled={!url || isValidating}>
                    {isValidating ? "Scanning…" : "Extract"}
                  </Button>
                </div>
                <span id="url-hint" className="text-sm text-muted-foreground">
                  Use a specific page (like /team, /about, or /author) for better results.
                </span>
              </div>
            </form>
          </CardContent>
        </Card>

        <Separator className="my-8" />

        <section aria-live="polite" aria-busy={isValidating} className="space-y-4">
          {error && <p className="text-sm text-red-600">Something went wrong. Please try again.</p>}
          {data && "ok" in data && !data.ok && <p className="text-sm text-red-600">{data.error}</p>}

          {isValidating && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="animate-pulse rounded-lg border p-4">
                  <div className="h-24 w-24 rounded-full bg-muted mb-4" />
                  <div className="h-4 bg-muted rounded w-3/4 mb-2" />
                  <div className="h-4 bg-muted rounded w-1/2 mb-6" />
                  <div className="h-3 bg-muted rounded w-full mb-2" />
                  <div className="h-3 bg-muted rounded w-5/6" />
                </div>
              ))}
            </div>
          )}

          {profiles && profiles.length > 0 && (
            <>
              <h2 className="text-xl font-semibold">Extracted profiles</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {profiles.map((p, idx) => (
                  <ProfileCard key={idx} profile={p} />
                ))}
              </div>
            </>
          )}

          {submitted && !isValidating && (!profiles || profiles.length === 0) && (
            <p className="text-sm text-muted-foreground">
              No profiles found on that page. Try a more specific page like a Team or Author page.
            </p>
          )}
        </section>
      </section>
    </main>
  )
}
