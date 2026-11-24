-- GradeAnt on Postgres (Supabase): one table per DynamoDB table.
--
-- Each row keeps the whole DynamoDB item in `item jsonb` (the source of truth), with the
-- key columns next to it. Index columns (DynamoDB GSIs) are GENERATED from `item`, so they
-- can never disagree with it. Rows without the attribute have NULL there and are left out
-- of `where col = $1`, exactly like DynamoDB's sparse indexes.
--
-- Applied by backend/scripts/sql-apply.mjs (production) and by the PGlite test driver.
-- Safe to re-run.

create schema if not exists gradeant;

create table if not exists gradeant.users (
  email text primary key,
  item  jsonb not null
);

create table if not exists gradeant.assignment_grading (
  evaluation_id text primary key,
  item          jsonb not null,
  student_id    text generated always as (item->>'student_id')    stored,
  "type"        text generated always as (item->>'type')          stored,
  assignment_id text generated always as (item->>'assignment_id') stored,
  qp_id         text generated always as (item->>'qp_id')         stored
);
create index if not exists grading_student_idx    on gradeant.assignment_grading (student_id);
create index if not exists grading_type_idx       on gradeant.assignment_grading ("type", assignment_id, qp_id);
create index if not exists grading_assignment_idx on gradeant.assignment_grading (assignment_id, qp_id);

create table if not exists gradeant.assignments (
  assignment_id text primary key,
  item          jsonb not null,
  teacher_id    text generated always as (item->>'teacher_id') stored
);
create index if not exists assignments_teacher_idx on gradeant.assignments (teacher_id);

create table if not exists gradeant.submissions (
  assignment_id text not null,
  student_id    text not null,
  item          jsonb not null,
  primary key (assignment_id, student_id)
);
create index if not exists submissions_student_idx on gradeant.submissions (student_id);

create table if not exists gradeant.announcements  (announcement_id text primary key, item jsonb not null);
create table if not exists gradeant.admin_users    (user_id         text primary key, item jsonb not null);
create table if not exists gradeant.reset_requests (request_id      text primary key, item jsonb not null);
create table if not exists gradeant.content        (post_id         text primary key, item jsonb not null);
create table if not exists gradeant.tickets        (ticket_id       text primary key, item jsonb not null);
create table if not exists gradeant.audit_logs     (log_id          text primary key, item jsonb not null);
create table if not exists gradeant.settings       (setting_key     text primary key, item jsonb not null);
create table if not exists gradeant.invites        ("token"         text primary key, item jsonb not null);
create table if not exists gradeant.analytics      (metric_key      text primary key, item jsonb not null);
