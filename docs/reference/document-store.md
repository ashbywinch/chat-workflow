# Document Store

Where a document lives is a property of the document, not of the component that
produced it, the process that caused it, or the person who ran it. This note
defines the store that records documents, the four operations the application
needs, and the simple implementation that ships first
(`document-store-files`).

**Status:** deferred design. The minimal path — meta-rules and message quality in
the existing feedback loop — is in
[the plan](../PLAN/standards-and-persistence.md). This page records the
persistence model for when that loop proves insufficient. Scope when built: one
user, local execution. The enterprise adapter is described in
[enterprise document management](enterprise-document-management.md) and is a
later implementation of the same interface.

## The problem

Business artefacts do not belong to the folder of whoever requested them.
Meeting minutes are kept by the meeting's owning role; brand guidelines are kept
by marketing even when a product team asked for them. A component's own
directory is therefore the wrong home for the artefact, and the session log is
not a store at all.

## Where it lives

**Persistence is not in the library.** The library produces values and raises
typed failures; it knows nothing about documents, storage or files. Persistence
is its own set of libraries, and the application (the component maker, the CLI)
chooses one and wires it in.

At least two implementations exist: a simple local one and a proper enterprise
one. They are separate libraries, not modes of one, and neither is "the"
persistence of the project. The contract they share is a third, tiny package;
one class per module, packages to organise them:

```
packages/document-store/src/document_store/
  document.py               # Document
  document_type.py          # DocumentType (StrEnum)
  document_store.py         # DocumentStore (Protocol)
  release.py                # Release
  errors.py                 # DocumentStoreError

packages/document-store-files/src/document_store_files/
  local_document_store.py   # LocalDocumentStore

packages/document-store-enterprise/   # later: object storage + metadata service
```

The library imports none of them; the adapters do not import the component
maker.

## The library-side interface

The integration is at **step boundaries**. The library returns typed results and
raises typed failures; the application persists them. Only two things are
needed on the library side, and neither is a store concept:

- **A rule will need identity.** In the deferred design, `_validation_rules`
  entries become `Rule(id, version, text)` values instead of bare strings, so a
  verdict can name the requirement version it operationalises.
- **The failure carries its evidence.** Today the judge returns typed verdicts
  and the library raises a `ValidationError` naming the rule and quoting the
  evidence; the model continues from that message. The deferred change is a
  verdict that also carries the rule id/version and the attempted artifact, so
  the application can persist and route it — one signal, no parallel report
  path.
- **No I/O.** The library never opens a file, never names a path, and never
  imports a store package.

The application persists at each step it owns: the requirements when they are
authored, the generated code, every artefact with the requirement versions it
was judged under, and every failure. The component maker's composites already
receive each child result, so this is explicit rather than a hook; whether to
add a thin `@persisted` wrapper is an application decision.

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
later adapter.

## The simple implementation

`document-store-files` writes `.workflow-store/`:

```
.workflow-store/            # configurable path
  index.jsonl               # append-only: versions and releases
  documents/<uuid>.json     # metadata, refs, small content
  blobs/<sha256>            # large content (files, PDFs, images)
```

It is a minimal DMS, not a database: append-only, inspectable, and safe to
commit or back up. Replacing it with object storage plus a metadata service is a
new implementation of the same contract, not a change to any caller.

## What is stored

- **Requirements**, with their intent, checkable statements, one good and one
  bad exemplar, scope, owner and versions.
- **Artefacts**, every version, with the requirement versions they were judged
  under. A release marks which versions matter; nothing is deleted.
- **Evaluations**, as documents that reference the artefact and the requirement
  versions.
- **Generated code**, as a document — and also as the runnable file the
  interpreter imports. The duplication is deliberate for now: the store is the
  record, the file is the executable.

## Seeding the eval of the eval

The rework loop is tested against a **real session from the failure that
motivated it**, truncated at the failure. The eval harness already writes that
session to the failure record (`<label>-<ts>.json`: `events` and the rendered
`transcript`, plus the verdicts). Curate one of those records, truncate it at
the failure, and replay its events through the rework step — no new capture
format. The router must reach the same conclusion on a conversation that
actually happened, which keeps the meta-eval faithful and deterministic.

## Enterprise direction

The physical/logical split, UUID identity, releases, metadata views, discovery
and document-level security are specified in
[enterprise document management](enterprise-document-management.md). The local
adapter implements the same four calls so that system is a replacement, not a
rewrite.

## Open questions

- Explicit saves at step boundaries versus a thin persistence wrapper in the
  component maker.
- Retention: nothing is deleted now; is that acceptable indefinitely?
- Blobs: local files first; which object store is the second adapter?
- Does the runnable code file live beside the store or inside it?
