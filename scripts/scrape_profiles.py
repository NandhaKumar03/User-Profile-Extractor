#!/usr/bin/env python3
"""
Profile Scraper - Extracts user profile data from websites
Usage: python scripts/scrape_profiles.py https://example.com/team
"""

import re
import sys
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('profile_scraper')

# Constants
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SOCIAL_PATTERNS = {
    "github": re.compile(r"github\.com/([^/\s]+)"),
    "linkedin": re.compile(r"linkedin\.com/(?:in|company)/([^/\s?]+)"),
    "twitter": re.compile(r"(?:twitter\.com|x\.com)/([^/\s?]+)"),
    "instagram": re.compile(r"instagram\.com/([^/\s?]+)"),
    "facebook": re.compile(r"facebook\.com/([^/\s?]+)"),
    "youtube": re.compile(r"youtube\.com/(?:user|channel)/([^/\s?]+)"),
    "medium": re.compile(r"medium\.com/@?([^/\s?]+)"),
    "scholar": re.compile(r"scholar\.google\.com/citations\?user=([^&\s]+)"),
}

class ProfileScraper:
    def __init__(self, url: str):
        self.url = url
        self.soup = None
        self.base_url = url
        self.profiles = []
        
    def fetch_page(self) -> bool:
        """Fetch the target page and prepare BeautifulSoup"""
        try:
            logger.info(f"Fetching {self.url}")
            response = requests.get(self.url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            
            if 'text/html' not in response.headers.get('Content-Type', ''):
                logger.error("URL did not return HTML content")
                return False
                
            self.soup = BeautifulSoup(response.text, 'html.parser')
            return True
        except Exception as e:
            logger.error(f"Error fetching page: {str(e)}")
            return False
    
    def clean_text(self, text: Optional[str]) -> Optional[str]:
        """Clean and normalize text"""
        if not text:
            return None
        return re.sub(r'\s+', ' ', text).strip() or None
    
    def extract_emails(self, text: str) -> List[str]:
        """Extract email addresses from text"""
        return [email.lower() for email in EMAIL_PATTERN.findall(text)]
    
    def extract_social_links(self, url: str) -> Dict[str, str]:
        """Extract social media information from URL"""
        result = {}
        for platform, pattern in SOCIAL_PATTERNS.items():
            match = pattern.search(url)
            if match:
                result = {
                    "platform": platform,
                    "username": match.group(1),
                    "url": url
                }
                break
        return result
    
    def resolve_url(self, href: Optional[str]) -> Optional[str]:
        """Resolve relative URLs to absolute URLs"""
        if not href:
            return None
        try:
            return urljoin(self.base_url, href)
        except:
            return None
    
    def extract_structured_data(self) -> List[Dict]:
        """Extract profile data from structured data (JSON-LD, microdata)"""
        profiles = []
        
        # JSON-LD extraction
        for script in self.soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                if not data:
                    continue
                    
                # Handle both single items and arrays
                items = data if isinstance(data, list) else [data]
                
                for item in items:
                    if not isinstance(item, dict):
                        continue
                        
                    # Check if this is a Person
                    item_type = item.get('@type')
                    if not item_type or 'person' not in str(item_type).lower():
                        continue
                        
                    # Extract profile data
                    profile = {
                        'name': self.clean_text(item.get('name')),
                        'title': self.clean_text(item.get('jobTitle')),
                        'email': self.clean_text(item.get('email')),
                        'bio': self.clean_text(item.get('description')),
                        'image': self.resolve_url(item.get('image', {}).get('url') if isinstance(item.get('image'), dict) else item.get('image')),
                        'socials': [],
                        'source': 'json-ld'
                    }
                    
                    # Extract social links
                    same_as = item.get('sameAs', [])
                    if isinstance(same_as, list):
                        for link in same_as:
                            url = self.resolve_url(link)
                            if url:
                                social = self.extract_social_links(url)
                                if social:
                                    profile['socials'].append(social)
                    
                    profiles.append(profile)
            except Exception as e:
                logger.debug(f"Error parsing JSON-LD: {str(e)}")
        
        # Microdata extraction
        for person in self.soup.select('[itemscope][itemtype*="Person"], [itemtype*="Person"]'):
            try:
                profile = {
                    'name': self.clean_text(person.select_one('[itemprop="name"]').get_text()) if person.select_one('[itemprop="name"]') else None,
                    'title': self.clean_text(person.select_one('[itemprop="jobTitle"]').get_text()) if person.select_one('[itemprop="jobTitle"]') else None,
                    'email': self.clean_text(person.select_one('[itemprop="email"]').get_text()) if person.select_one('[itemprop="email"]') else None,
                    'bio': self.clean_text(person.select_one('[itemprop="description"]').get_text()) if person.select_one('[itemprop="description"]') else None,
                    'image': self.resolve_url(person.select_one('[itemprop="image"]').get('src')) if person.select_one('[itemprop="image"]') else None,
                    'socials': [],
                    'source': 'microdata'
                }
                
                # Extract social links
                for link in person.select('[itemprop="sameAs"], [itemprop="url"]'):
                    url = self.resolve_url(link.get('href'))
                    if url:
                        social = self.extract_social_links(url)
                        if social:
                            profile['socials'].append(social)
                
                profiles.append(profile)
            except Exception as e:
                logger.debug(f"Error parsing microdata: {str(e)}")
        
        return profiles
    
    def extract_wikipedia_profiles(self) -> List[Dict]:
        """Extract profile data from Wikipedia pages"""
        if 'wikipedia.org' not in self.url and not self.soup.select('.mw-parser-output'):
            return []
            
        profiles = []
        
        try:
            # Get name from page title
            name = self.clean_text(self.soup.select_one('h1#firstHeading').get_text()) if self.soup.select_one('h1#firstHeading') else None
            
            # Get image from infobox
            infobox = self.soup.select_one('.infobox, .biography, .vcard')
            image = None
            if infobox and infobox.select_one('img'):
                img_src = infobox.select_one('img').get('src')
                image = self.resolve_url(img_src)
                
                # Fix Wikipedia thumbnail URLs
                if image and '/thumb/' in image:
                    parts = image.split('/')
                    if 'thumb' in parts:
                        thumb_index = parts.index('thumb')
                        if thumb_index < len(parts) - 2:
                            filename = parts[-1]
                            size_match = re.search(r'(\d+)px-', filename)
                            if size_match:
                                new_filename = filename[size_match.end():]
                                parts = parts[:thumb_index] + parts[thumb_index+1:-1] + [new_filename]
                                image = '/'.join(parts)
            
            # Get bio from first paragraph
            bio = None
            first_p = self.soup.select_one('.mw-parser-output > p')
            if first_p:
                bio = self.clean_text(first_p.get_text())
            
            # Get occupation/title
            occupation = None
            if infobox:
                for row in infobox.select('tr'):
                    header = row.select_one('th')
                    if header and any(term in header.get_text().lower() for term in ['occupation', 'profession']):
                        occupation = self.clean_text(row.select_one('td').get_text()) if row.select_one('td') else None
                        break
            
            # Extract external links
            socials = []
            external_links = self.soup.select_one('#External_links, #External_sites')
            if external_links:
                link_list = external_links.find_next('ul')
                if link_list:
                    for link in link_list.select('a[href]'):
                        url = self.resolve_url(link.get('href'))
                        if url:
                            social = self.extract_social_links(url)
                            if social:
                                socials.append(social)
            
            if name or bio or image:
                profiles.append({
                    'name': name,
                    'title': occupation,
                    'email': None,  # Wikipedia typically doesn't include emails
                    'bio': bio,
                    'image': image,
                    'socials': socials,
                    'source': 'wikipedia'
                })
                
        except Exception as e:
            logger.debug(f"Error extracting Wikipedia profile: {str(e)}")
            
        return profiles
    
    def extract_common_profiles(self) -> List[Dict]:
        """Extract profiles using common patterns and heuristics"""
        profiles = []
        
        # Look for profile containers
        profile_selectors = [
            '.team-member', '.profile', '.person', '.staff-member', '.faculty-member',
            '.author', '.bio', '[class*="profile"]', '[class*="team"]', '[class*="member"]',
            '[class*="person"]', '[class*="author"]', '.vcard', '.h-card'
        ]
        
        profile_elements = []
        for selector in profile_selectors:
            profile_elements.extend(self.soup.select(selector))
        
        # If no profile containers found, try to find tables that might contain profiles
        if not profile_elements:
            tables = self.soup.select('table')
            for table in tables:
                if table.select('img') and any(term in str(table).lower() for term in ['name', 'team', 'staff', 'member']):
                    for row in table.select('tr'):
                        if row.select('td') and not (row.select('th') and not row.select('td')):
                            profile_elements.append(row)
        
        # Process each profile element
        for element in profile_elements:
            try:
                # Extract name
                name = None
                name_el = element.select_one('h1, h2, h3, h4, .name, [class*="name"]')
                if name_el:
                    name = self.clean_text(name_el.get_text())
                
                # Extract title/position
                title = None
                title_el = element.select_one('.title, .position, .role, [class*="title"], [class*="position"], [class*="role"]')
                if title_el:
                    title = self.clean_text(title_el.get_text())
                
                # Extract image
                image = None
                img_el = element.select_one('img')
                if img_el:
                    image = self.resolve_url(img_el.get('src') or img_el.get('data-src'))
                
                # Extract bio
                bio = None
                bio_el = element.select_one('p, .bio, .description, [class*="bio"], [class*="description"]')
                if bio_el:
                    bio = self.clean_text(bio_el.get_text())
                
                # Extract email
                email = None
                # Check for mailto links
                mailto = element.select_one('a[href^="mailto:"]')
                if mailto:
                    email_match = EMAIL_PATTERN.search(mailto.get('href', ''))
                    if email_match:
                        email = email_match.group(0).lower()
                
                # If no email found, look in text
                if not email:
                    emails = self.extract_emails(element.get_text())
                    if emails:
                        email = emails[0]
                
                # Extract social links
                socials = []
                for link in element.select('a[href]'):
                    url = self.resolve_url(link.get('href'))
                    if url:
                        social = self.extract_social_links(url)
                        if social:
                            socials.append(social)
                
                # Only add if we found meaningful data
                if name or title or email or image or bio or socials:
                    profiles.append({
                        'name': name,
                        'title': title,
                        'email': email,
                        'bio': bio,
                        'image': image,
                        'socials': socials,
                        'source': 'heuristic'
                    })
            except Exception as e:
                logger.debug(f"Error extracting profile from element: {str(e)}")
        
        return profiles
    
    def merge_profiles(self, profiles: List[Dict]) -> List[Dict]:
        """Merge duplicate profiles based on name or other identifiers"""
        if not profiles:
            return []
            
        merged = []
        for profile in profiles:
            # Skip empty profiles
            if not any(profile.get(field) for field in ['name', 'email', 'image', 'bio']):
                continue
                
            # Check if this profile should be merged with an existing one
            found_match = False
            for existing in merged:
                # Check for matching name (case insensitive)
                same_name = (profile.get('name') and existing.get('name') and 
                            profile['name'].lower() == existing['name'].lower())
                
                # Check for matching email
                same_email = (profile.get('email') and existing.get('email') and 
                             profile['email'].lower() == existing['email'].lower())
                
                # Check for overlapping social profiles
                profile_socials = {s.get('url') for s in profile.get('socials', []) if s.get('url')}
                existing_socials = {s.get('url') for s in existing.get('socials', []) if s.get('url')}
                overlapping_socials = profile_socials.intersection(existing_socials)
                
                if same_name or same_email or overlapping_socials:
                    # Merge the profiles
                    existing['name'] = existing.get('name') or profile.get('name')
                    existing['title'] = existing.get('title') or profile.get('title')
                    existing['email'] = existing.get('email') or profile.get('email')
                    existing['bio'] = existing.get('bio') or profile.get('bio')
                    existing['image'] = existing.get('image') or profile.get('image')
                    
                    # Merge socials without duplicates
                    all_socials = existing.get('socials', []) + profile.get('socials', [])
                    unique_socials = []
                    seen_urls = set()
                    for social in all_socials:
                        if social.get('url') and social['url'] not in seen_urls:
                            unique_socials.append(social)
                            seen_urls.add(social['url'])
                    existing['socials'] = unique_socials
                    
                    # Track sources
                    sources = set([existing.get('source', ''), profile.get('source', '')])
                    existing['source'] = ', '.join(filter(None, sources))
                    
                    found_match = True
                    break
            
            if not found_match:
                merged.append(profile)
        
        return merged
    
    def scrape(self) -> Dict:
        """Main method to scrape profiles from the URL"""
        if not self.fetch_page():
            return {"ok": False, "error": "Failed to fetch page"}
        
        try:
            # Extract profiles using different methods
            structured_profiles = self.extract_structured_data()
            wikipedia_profiles = self.extract_wikipedia_profiles()
            common_profiles = self.extract_common_profiles()
            
            # Combine all profiles
            all_profiles = structured_profiles + wikipedia_profiles + common_profiles
            
            # Merge duplicate profiles
            merged_profiles = self.merge_profiles(all_profiles)
            
            # Clean up profiles - remove None values for cleaner output
            for profile in merged_profiles:
                for key in list(profile.keys()):
                    if profile[key] is None:
                        del profile[key]
            
            return {
                "ok": True,
                "url": self.url,
                "profiles": merged_profiles
            }
            
        except Exception as e:
            logger.error(f"Error during scraping: {str(e)}")
            return {"ok": False, "error": str(e)}

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "Usage: python scripts/scrape_profiles.py <url>"}))
        sys.exit(1)
    
    url = sys.argv[1]
    
    # Validate URL
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            print(json.dumps({"ok": False, "error": "Invalid URL format"}))
            sys.exit(1)
    except:
        print(json.dumps({"ok": False, "error": "Invalid URL"}))
        sys.exit(1)
    
    # Run the scraper
    t0 = time.time()
    scraper = ProfileScraper(url)
    result = scraper.scrape()
    result["_t_ms"] = int((time.time() - t0) * 1000)
    
    # Output results
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()