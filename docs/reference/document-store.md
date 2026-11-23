# Document Store

Where a document lives is a property of the document, not of the component that
produced it, the process that caused it, or the person who ran it. This note
defines the store that records documents, the four operations the framework
needs, and the local adapter that ships first.

**Status:** proposal. Scope: single user, local execution. The enterprise
adapter (object storage + DMS + search) is a later implementation of the same
interface.

## The problem

Business artefacts do not belong to the folder of whoever requested them.
Meeting minutes are kept by the meeting's owning role; brand guidelines are kept
by marketing even when a product team asked for them. A component's own
directory is therefore the wrong home for the artefact, and the session log is
not a store at all.

The current state is thinner than that: a generated module is written to
`workflows/<name>/`, the diagram goes to a temporary directory, and the
requirements, artefacts and verdicts exist only in transcripts.

## One store, one record

There is **one** store. Requirements, artefacts, evaluations and generated code
are all documents, distinguished by `type` and metadata. They are not separate
stores; a separate store per document type reintroduces the folder hierarchy
under another name.

A record is the metadata schema an enterprise DMS would hold, so an adapter can
replace the local files without changing the model:

```json
{
  "uuid": "…",
  "type": "requirement | artifact | evaluation | code",
  "title": "…",
  "owning_role": "…",
  "status": "draft | final | superseded",
  "refs": ["uuid", "…"],
  "content": "…",          // small text, or
  "blob": "sha256:…",      // large content in blobs/
  "version": 1,
  "created_at": "…",
  "source": {"component_uuid": "…", "component_version": 3},
  "requirement_versions": [{"id": "…", "version": 2}],
  "release": null
}
```

**`owning_role` is the role the document belongs to**, not the role that wrote
it. For minutes it is the meeting's owning role — which depends on the meeting
and is usually not the note-taker. The value is resolved per document when it is
created; it is not a fixed owner for a document type.

**Composition is by reference.** A document carries `refs` to its parts; parts
may be nested arbitrarily deep, may be shared by several parents, and there is
no single top level. A "view" is a query, not a root object.

## Four operations

- `put(document) -> uuid@version`
- `get(uuid, version=None)`
- `find(type=None, owning_role=None, release=None)`
- `release(name, [uuid@version])` — an immutable snapshot of the named versions

Everything else — folders, role views, search, permissions — is a query or a
later adapter. The workflow needs only these four.

## The local adapter

```
.workflow-store/            # configurable path
  index.jsonl               # append-only: versions and releases
  documents/<uuid>.json     # metadata, refs, small content
  blobs/<sha256>            # large content (files, PDFs, images)
```

It is a minimal DMS, not a database: append-only, inspectable, and safe to
commit or back up. Replacing it with object storage plus a metadata service is a
new adapter, not a change to any caller.

## What is stored

- **Requirements**, with their versions and the checks that operationalise them.
- **Artefacts**, every version, with the requirement versions they were judged
  under. A release marks which versions matter; nothing is deleted.
- **Evaluations**, as documents that reference the artefact and the requirement
  versions.
- **Generated code**, as a document — and also as the runnable file the
  interpreter imports. The duplication is deliberate for now: the store is the
  record, the file is the executable.

## Independence

The store lives in the core package (`packages/chat-workflow`). It knows only
documents; it does not import the component maker, and it holds no server, no
database and no vendor SDK. The component maker depends on it, one way.

The library still owns no persistence in the PRD's sense: this is an optional
local adapter behind a port, not a service. That sentence in the PRD needs
amending to say so.

## Open questions

- Retention: nothing is deleted now; is that acceptable indefinitely?
- Blobs: local files first; which object store is the second adapter?
- Does the runnable code file live beside the store or inside it, once the
  component maker is split out?
