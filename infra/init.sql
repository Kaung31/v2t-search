CREATE TABLE assets (
  id              BIGSERIAL PRIMARY KEY,
  content_hash    TEXT UNIQUE NOT NULL,    -- idempotency key
  observation_id  TEXT NOT NULL,            -- Brain4Cars id
  inside_path     TEXT,
  outside_path    TEXT,
  gps_path        TEXT,
  maneuver_label  TEXT,
  duration_s      REAL,
  created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE frames (
  id              BIGSERIAL PRIMARY KEY,
  asset_id        BIGINT REFERENCES assets(id),
  camera          TEXT NOT NULL,            -- 'inside' | 'outside'
  frame_index     INT NOT NULL,
  timestamp_s     REAL NOT NULL,
  thumbnail_path  TEXT,
  status          TEXT NOT NULL,
  UNIQUE (asset_id, camera, frame_index)
);

CREATE TABLE events (
  id              BIGSERIAL PRIMARY KEY,
  asset_id        BIGINT REFERENCES assets(id),
  camera          TEXT NOT NULL,            -- or 'kinematic' for GPS-derived
  event_type      TEXT NOT NULL,
  start_ts        REAL NOT NULL,
  end_ts          REAL NOT NULL,
  magnitude       REAL,
  payload         JSONB
);
CREATE INDEX events_asset_cam_time ON events(asset_id, camera, start_ts);
CREATE INDEX events_asset_type_time ON events(asset_id, event_type, start_ts);

CREATE TABLE feedback (
  id              BIGSERIAL PRIMARY KEY,
  query_id        TEXT NOT NULL,
  query_text      TEXT NOT NULL,
  result_id       TEXT NOT NULL,
  vote            SMALLINT NOT NULL,        -- +1, -1
  clicked         BOOLEAN NOT NULL,
  layer_scores    JSONB NOT NULL,
  created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE eval_runs (
  id              BIGSERIAL PRIMARY KEY,
  run_at          TIMESTAMPTZ DEFAULT now(),
  git_sha         TEXT,
  config_name     TEXT,                     -- 'semantic_only', 'full', etc.
  metrics         JSONB
);