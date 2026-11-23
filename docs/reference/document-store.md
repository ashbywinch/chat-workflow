# Document Store

Where a document lives is a property of the document, not of the component that
produced it, the process that caused it, or the person who ran it. This note
defines the store that records documents, the four operations the application
needs, and the local adapter that ships first.

**Status:** proposal. Scope: single user, local execution. The enterprise
adapter is described in
[enterprise document management](enterprise-document-management.md) and is a
later implementation of the same interface.

## The problem

Business artefacts do not belong to the folder of whoever requested them.
Meeting minutes are kept by the meeting's owning role; brand guidelines are kept
by marketing even when a product team asked for them. A component's own
directory is therefore the wrong home for the artefact, and the session log is
not a store at all.

## Where it lives

**Persistence is not in the library.** The library produces values and knows
nothing about documents, storage or files. A separate persistence package owns
the store, and the application (the component maker, the CLI) wires it in.

One class per module, packages to organise them:

```
packages/chat-workflow-persistence/src/chat_workflow_persistence/
  document.py               # Document
  document_type.py          # DocumentType (StrEnum)
  document_store.py         # DocumentStore (Protocol)
  local_document_store.py   # LocalDocumentStore
  release.py                # Release
  errors.py                 # DocumentStoreError
```

The library does not import this package; the package does not import the
component maker.

## The library-side interface

The integration is at **step boundaries**. The library returns typed results;
the application persists them. Three things make that interface neat, and only
the first is a library change:

- **A verdict is a value, not an exception.** `LLMValidated` currently raises
  inside a model validator, so a semantic verdict exists only as an error. The
  refactor splits it: structural validation still raises (a malformed object is
  returned to the model); semantic evaluation returns a `ValidationReport` of
  verdicts, each naming the rule id and version and carrying its evidence. The
  report rides on `TurnResult`/`SessionLog`, where the application can read it
  and persist it. The existing raising behaviour remains as a wrapper for callers
  that want it.
- **A rule has identity.** `_validation_rules` entries become
  `Rule(id, version, text)` values instead of bare strings, so a verdict can be
  tied to the requirement it operationalises. This is a value type in the
  library, not a store concept.
- **No I/O.** The library never opens a file, never names a path, and never
  imports the persistence package.

The application persists at each step it owns: the requirements when they are
authored, the generated code, every artefact with the requirement versions it
was judged under, and every evaluation report. The component maker's composites
already receive each child result, so this is explicit rather than a hook;
whether to add a thin `@persisted` wrapper is an application decision.

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

## Seeding the eval of the eval

The rework loop is tested against a **real session from the failure that
motivated it**, truncated at the failure, not a synthetic one. Capture the raw
PydanticAI messages (`session_logging.dump`) when an eval fails, curate one into
a fixture, and replay it through the rework step: the router must reach the same
conclusion on a conversation that actually happened. This keeps the meta-eval
faithful and deterministic without inventing a scenario.

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
