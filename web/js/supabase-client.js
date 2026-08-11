function getSupabaseConfig() {
  var cfg = window.APP_CONFIG || {};
  if (!cfg.SUPABASE_URL || !cfg.SUPABASE_ANON_KEY) {
    throw new Error(
      "Missing Supabase config. Set SUPABASE_URL and SUPABASE_ANON_KEY in web/config.js"
    );
  }
  return cfg;
}

function createSupabaseClient() {
  var cfg = getSupabaseConfig();
  return window.supabase.createClient(cfg.SUPABASE_URL, cfg.SUPABASE_ANON_KEY);
}

window.createSupabaseClient = createSupabaseClient;
