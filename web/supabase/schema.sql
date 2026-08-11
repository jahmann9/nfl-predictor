-- Run this in your Supabase SQL editor.

create extension if not exists pgcrypto;

create table if not exists public.weekly_picks (
  id uuid primary key default gen_random_uuid(),
  season integer not null check (season >= 2020),
  week integer not null check (week between 1 and 22),
  person text not null check (person in ('Jaron', 'Tom', 'Dylan', 'Jordan', 'Jacob')),
  pick_type text not null check (pick_type in ('spread', 'ou')),
  pick_text text not null,
  result text not null default 'pending' check (result in ('pending', 'hit', 'miss')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (season, week, person)
);

create or replace function public.set_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists trg_weekly_picks_updated_at on public.weekly_picks;
create trigger trg_weekly_picks_updated_at
before update on public.weekly_picks
for each row execute function public.set_updated_at();

alter table public.weekly_picks enable row level security;

-- Public can view leaderboard and picks.
drop policy if exists weekly_picks_public_read on public.weekly_picks;
create policy weekly_picks_public_read
on public.weekly_picks
for select
using (true);

-- Only authenticated users can edit (use admin creds for access).
drop policy if exists weekly_picks_auth_write on public.weekly_picks;
create policy weekly_picks_auth_write
on public.weekly_picks
for all
using (auth.role() = 'authenticated')
with check (auth.role() = 'authenticated');
