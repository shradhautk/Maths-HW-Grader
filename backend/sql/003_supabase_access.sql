-- Supabase only (uses Supabase's roles; the PGlite test database skips this file).
--
-- The browser never talks to the database: the web app calls the Node API, which connects
-- as the `postgres` role through the pooler. RLS is on with NO policies, so the public
-- `anon` key and logged-in Supabase users (`authenticated`) can read and write nothing.
--
-- The Python grader Lambda uses PostgREST with the `service_role` key and may touch only
-- what it needs: select on assignment_grading and the two functions in 002.
-- Dashboard step (once): Project Settings → API → Exposed schemas: add `gradeant`.

do $$
declare t text;
begin
  for t in select tablename from pg_tables where schemaname = 'gradeant' loop
    execute format('alter table gradeant.%I enable row level security', t);
  end loop;
end
$$;

revoke all on schema gradeant from public, anon, authenticated;
revoke all on all tables in schema gradeant from public, anon, authenticated;
revoke all on all functions in schema gradeant from public, anon, authenticated;
alter default privileges in schema gradeant revoke all on tables from public, anon, authenticated;
alter default privileges in schema gradeant revoke all on functions from public, anon, authenticated;

grant usage on schema gradeant to service_role;
grant select on gradeant.assignment_grading to service_role;
grant execute on function gradeant.grading_replace_rows(jsonb, text[]) to service_role;
grant execute on function gradeant.grading_patch(text, jsonb, text[]) to service_role;
