-- Functions the Python grading Lambda calls through Supabase's REST API (PostgREST RPC).
-- PostgREST has no multi-statement transactions, so each write the grader needs is one
-- function = one transaction. Who may call them is in 003_supabase_access.sql.

-- Replace/insert grading rows and delete stale ones in ONE transaction
-- (python-backend gradeant_common.grading.replace_rows; the summary row goes in a second call, last).
create or replace function gradeant.grading_replace_rows(p_rows jsonb, p_delete text[] default '{}')
returns integer
language plpgsql
security definer  -- runs as the owner: callers need only EXECUTE (003), never table write rights
set search_path = ''
as $$
begin
  if jsonb_typeof(p_rows) <> 'array' then
    raise exception 'p_rows must be a JSON array';
  end if;
  delete from gradeant.assignment_grading where evaluation_id = any(p_delete);
  insert into gradeant.assignment_grading (evaluation_id, item)
    select r->>'evaluation_id', r from jsonb_array_elements(p_rows) as r
  on conflict (evaluation_id) do update set item = excluded.item;
  return jsonb_array_length(p_rows);
end
$$;

-- DynamoDB "SET a = :v, … REMOVE x, y" without a condition: an upsert, like UpdateItem
-- (python-backend gradeant_common.grading.patch_row; used by the regrade path).
create or replace function gradeant.grading_patch(p_id text, p_set jsonb, p_remove text[] default '{}')
returns void
language sql
security definer  -- runs as the owner: callers need only EXECUTE (003), never table write rights
set search_path = ''
as $$
  insert into gradeant.assignment_grading (evaluation_id, item)
    values (p_id, (jsonb_build_object('evaluation_id', p_id) || p_set) - p_remove)
  on conflict (evaluation_id) do update
    set item = (gradeant.assignment_grading.item || p_set) - p_remove;
$$;
