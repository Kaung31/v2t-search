CREATE TABLE assets (
  id              BIGSERIAL PRIMARY KEY,
  content_hash    TEXT UNIQUE NOT NULL,
  sample_id       TEXT NOT NULL,
  split           TEXT NOT NULL,
  duration_s      REAL DEFAULT 3.0,
  created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE clips (
  id              BIGSERIAL PRIMARY KEY,
  asset_id        BIGINT REFERENCES assets(id),
  view            TEXT NOT NULL,
  video_path      TEXT,
  frames_dir      TEXT,
  status          TEXT NOT NULL DEFAULT 'pending',
  UNIQUE (asset_id, view)
);

CREATE TABLE frames (
  id              BIGSERIAL PRIMARY KEY,
  clip_id         BIGINT REFERENCES clips(id),
  frame_index     INT NOT NULL,
  timestamp_s     REAL NOT NULL,
  thumbnail_path  TEXT,
  UNIQUE (clip_id, frame_index)
);

CREATE TABLE events (
  id              BIGSERIAL PRIMARY KEY,
  asset_id        BIGINT REFERENCES assets(id),
  view            TEXT NOT NULL,
  event_type      TEXT NOT NULL,
  source          TEXT NOT NULL,
  start_ts        REAL NOT NULL,
  end_ts          REAL NOT NULL,
  magnitude       REAL,
  payload         JSONB
);
CREATE INDEX events_asset_view_time ON events(asset_id, view, start_ts);
CREATE INDEX events_asset_type_time ON events(asset_id, event_type, start_ts);
CREATE INDEX events_asset_source    ON events(asset_id, source);

CREATE TABLE feedback (
  id              BIGSERIAL PRIMARY KEY,
  query_id        TEXT NOT NULL,
  query_text      TEXT NOT NULL,
  sample_id       TEXT NOT NULL,
  vote            SMALLINT NOT NULL,
  clicked         BOOLEAN NOT NULL,
  layer_scores    JSONB NOT NULL,
  created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE eval_runs (
  id              BIGSERIAL PRIMARY KEY,
  run_at          TIMESTAMPTZ DEFAULT now(),
  git_sha         TEXT,
  config_name     TEXT,
  metrics         JSONB
);