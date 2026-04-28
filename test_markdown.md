# Extended Syntax

Advanced features that build on the basic Markdown syntax.

## Overview

The basic syntax outlined in the original Markdown design document added many of the elements needed on a day-to-day basis, but it wasn't enough for some people. That's where extended syntax comes in.

Several individuals and organizations took it upon themselves to extend the basic syntax by adding additional elements like **tables**, code blocks, syntax highlighting, URL auto-linking, and footnotes.

## Availability

Not all Markdown applications support extended syntax elements.

### Lightweight Markup Languages

- CommonMark
- GitHub Flavored Markdown (GFM)
- Markdown Extra
- MultiMarkdown
- R Markdown

## Tables

To add a table, use three or more hyphens (`---`) to create each column header, and use pipes (`|`) to separate each column.

| Syntax    | Description | Example     |
| --------- | ----------- | ----------- |
| Header    | Title       | # Heading   |
| Paragraph | Text        | Plain text  |
| Bold      | Strong      | **bold**    |
| Italic    | Emphasis    | *italic*    |
| Code      | Inline code | `code`     |

### Alignment

You can align text in columns using colons (`:`) in the header row.

| Left-aligned | Center-aligned | Right-aligned |
| :----------- | :------------: | ------------: |
| git status   | git status     | git status    |
| git diff     | git diff       | git diff      |
| git commit   | git commit     | git commit    |

## Cloud Provider Pricing

Approximate on-demand pricing per vCPU-hour as of 2025:

| Provider     | Instance Type  | vCPUs | RAM (GB) | Price (USD) | Region       |
| ------------ | -------------- | ----- | -------- | ----------- | ------------ |
| AWS          | t3.medium      | 2     | 4        | $0.0416    | us-east-1    |
| GCP          | e2-medium      | 2     | 4        | $0.0335    | us-central1  |
| Azure        | B2s            | 2     | 4        | $0.0496    | eastus       |
| Hetzner      | CX21           | 2     | 4        | $0.0083    | eu-central   |
| DigitalOcean | s-2vcpu-4gb    | 2     | 4        | $0.0298    | nyc1         |

> **Note:** Prices vary by commitment level (on-demand vs reserved). Always verify on official pricing pages.

## Fenced Code Blocks

Use triple backticks to create fenced code blocks:

```json
{
  firstName: John,
  lastName: Smith,
  age: 25,
  languages: [Python, Rust, Go]
}
```

### Python Example

```python
def fibonacci(n: int) -> list[int]:
    a, b = 0, 1
    result = []
    for _ in range(n):
        result.append(a)
        a, b = b, a + b
    return result

print(fibonacci(10))
```

## ML Model Comparison

| Model             | Params | MMLU (%) | HumanEval (%) | Context (tokens) | Open Source |
| ----------------- | ------ | -------- | ------------- | ---------------- | ----------- |
| GPT-4o            | ~200B  | 88.7     | 90.2          | 128k             | No          |
| Claude 3.5 Sonnet | ~70B   | 88.3     | 92.0          | 200k             | No          |
| Llama 3.1 70B     | 70B    | 82.6     | 80.5          | 128k             | Yes         |
| Mistral Large     | 123B   | 81.2     | 45.1          | 128k             | No          |
| Qwen2.5 72B       | 72B    | 86.0     | 86.6          | 128k             | Yes         |

## Task Lists

- [x] Write the press release
- [x] Update the website
- [ ] Contact the media
- [ ] Deploy to production

## Strikethrough and Emphasis

~~The world is flat.~~ We now know that the world is round.

This text has **bold**, *italic*, ***bold italic***, and `inline code`.

## Blockquotes

> Dorothy followed her through many rooms.
>
> The Witch bade her clean the pots and kettles and sweep the floor and keep the fire fed with wood.

## Horizontal Rule

---

## Footnotes

Here is a simple footnote.[^1] And here is a longer one.[^bignote]

[^1]: This is the first footnote.
[^bignote]: Here is one with multiple paragraphs and code. `{ my code }`

## Automatic URL Linking

Visit https://www.markdownguide.org for more info.
