import sqlite3
conn = sqlite3.connect('data/obs/observations.db')
print('Total obs:', conn.execute('SELECT count(*) FROM daily_obs').fetchone())
print('CWA BOU stations:', conn.execute("SELECT count(*) FROM stations WHERE cwa='BOU'").fetchone())
print('Sample stids in BOU:', conn.execute("SELECT stid FROM stations WHERE cwa='BOU' LIMIT 5").fetchall())
print('Sample obs rows:', conn.execute('SELECT stid, date, maxt_f, mint_f FROM daily_obs LIMIT 5').fetchall())

bou_stids = [r[0] for r in conn.execute("SELECT stid FROM stations WHERE cwa='BOU'").fetchall()]
if bou_stids:
    ph = ','.join('?' for _ in bou_stids)
    rows = conn.execute(
        f"SELECT stid, date, maxt_f, maxt_obs_count, mint_f, mint_obs_count "
        f"FROM daily_obs WHERE stid IN ({ph}) AND date BETWEEN '2025-10-01' AND '2025-10-04' ORDER BY date, stid LIMIT 15",
        bou_stids
    ).fetchall()
    print('\nDEN-area Oct 1-4 sample:')
    for r in rows:
        print(r)
    rng = conn.execute(
        f"SELECT min(maxt_f), max(maxt_f), min(mint_f), max(mint_f) FROM daily_obs WHERE stid IN ({ph})",
        bou_stids
    ).fetchone()
    print(f"\nBOU MaxT range: {rng[0]:.1f} - {rng[1]:.1f} F")
    print(f"BOU MinT range: {rng[2]:.1f} - {rng[3]:.1f} F")
conn.close()
