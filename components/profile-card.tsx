import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Mail, LinkIcon, Github, Linkedin, Twitter, Globe } from "lucide-react"

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

const SocialIcon = ({ type }: { type: string | null | undefined }) => {
  if (!type) {
    return null; // Or a default icon, or handle as appropriate
  }
  const t = type.toLowerCase();
  if (t.includes("github")) return <Github className="h-4 w-4" aria-hidden />;
  if (t.includes("linkedin")) return <Linkedin className="h-4 w-4" aria-hidden />;
  if (t.includes("twitter") || t.includes("x")) return <Twitter className="h-4 w-4" aria-hidden />;
  if (t.includes("website") || t.includes("site") || t.includes("web")) return <Globe className="h-4 w-4" aria-hidden />
  return <LinkIcon className="h-4 w-4" aria-hidden />
}

export function ProfileCard({ profile }: { profile: Profile }) {
  const { name, title, email, image, bio, socials = [], links = [], source } = profile

  return (
    <Card className="overflow-hidden">
      <CardHeader className="flex flex-row items-center gap-4">
        <img
          src={image || "/placeholder.svg?height=96&width=96&query=profile%20avatar"}
          alt={name ? `${name}'s avatar` : "Profile avatar"}
          className="h-16 w-16 rounded-full object-cover border"
        />
        <div className="min-w-0">
          <CardTitle className="truncate">{name || "Unknown name"}</CardTitle>
          {title && <p className="text-sm text-muted-foreground">{title}</p>}
          {email && (
            <p className="text-sm mt-1 inline-flex items-center gap-1">
              <Mail className="h-4 w-4" aria-hidden />
              <a className="underline underline-offset-2" href={`mailto:${email}`}>
                {email}
              </a>
            </p>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {bio && <p className="text-sm leading-relaxed">{bio}</p>}

        {(socials.length > 0 || links.length > 0) && (
          <div className="space-y-2">
            {socials.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {socials.map((s, i) => (
                  <a
                    key={i}
                    className="inline-flex items-center gap-1 rounded border px-2 py-1 text-xs hover:bg-accent"
                    href={s.url}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <SocialIcon type={s.type} />
                    <span className="truncate max-w-[160px]">{s.type}</span>
                  </a>
                ))}
              </div>
            )}
            {links.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {links.slice(0, 4).map((l, i) => (
                  <a
                    key={i}
                    className="inline-flex items-center gap-1 rounded border px-2 py-1 text-xs hover:bg-accent"
                    href={l}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <LinkIcon className="h-4 w-4" aria-hidden />
                    <span className="truncate max-w-[200px]">{l}</span>
                  </a>
                ))}
              </div>
            )}
          </div>
        )}

        {source && (
          <div>
            <Badge variant="secondary" aria-label={`source ${source}`}>
              source: {source}
            </Badge>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
