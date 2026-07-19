/**
 * Keyless Open-Meteo fallback for per-gu weather and PM2.5 overlays.
 *
 * Centroids reuse the existing Seoul 25-gu coordinates in
 * ``components/Map3D.tsx``. They live here as a provider-local lookup
 * because importing the client map component into an Edge route would
 * pull browser-only rendering dependencies into the server bundle.
 */

interface GuCentroid {
  lat: number;
  lon: number;
}

const GU_CENTROIDS: Record<string, GuCentroid> = {
  강남구: { lat: 37.5172, lon: 127.0473 },
  강동구: { lat: 37.5301, lon: 127.1238 },
  강북구: { lat: 37.6396, lon: 127.0257 },
  강서구: { lat: 37.5509, lon: 126.8495 },
  관악구: { lat: 37.4783, lon: 126.9514 },
  광진구: { lat: 37.5384, lon: 127.0823 },
  구로구: { lat: 37.4954, lon: 126.8874 },
  금천구: { lat: 37.4519, lon: 126.8954 },
  노원구: { lat: 37.6543, lon: 127.0566 },
  도봉구: { lat: 37.6688, lon: 127.0471 },
  동대문구: { lat: 37.5744, lon: 127.0398 },
  동작구: { lat: 37.5124, lon: 126.9393 },
  마포구: { lat: 37.5664, lon: 126.9015 },
  서대문구: { lat: 37.5793, lon: 126.9367 },
  서초구: { lat: 37.4837, lon: 127.0327 },
  성동구: { lat: 37.5634, lon: 127.0367 },
  성북구: { lat: 37.5894, lon: 127.0167 },
  송파구: { lat: 37.5145, lon: 127.1058 },
  양천구: { lat: 37.5169, lon: 126.8666 },
  영등포구: { lat: 37.5263, lon: 126.8956 },
  용산구: { lat: 37.5326, lon: 126.9788 },
  은평구: { lat: 37.6027, lon: 126.9292 },
  종로구: { lat: 37.5735, lon: 126.9794 },
  중구: { lat: 37.5636, lon: 126.9974 },
  중랑구: { lat: 37.6063, lon: 127.0925 },
};

interface OpenMeteoWeatherResponse {
  current?: {
    temperature_2m?: number;
    relative_humidity_2m?: number;
  };
}

interface OpenMeteoAirResponse {
  current?: {
    pm2_5?: number;
  };
}

export interface OpenMeteoWeatherRow {
  gu_nm: string;
  temp: number;
  humidity: number;
}

export interface OpenMeteoAirRow {
  gu_nm: string;
  pm25: number;
}

export async function fetchOpenMeteoWeather(
  gus: readonly string[],
): Promise<OpenMeteoWeatherRow[] | null> {
  try {
    const coords = coordinatesFor(gus);
    const payload = await fetchBatch<OpenMeteoWeatherResponse>(
      "https://api.open-meteo.com/v1/forecast",
      coords,
      "temperature_2m,relative_humidity_2m",
    );
    return gus.map((gu, i) => {
      const temp = payload[i]?.current?.temperature_2m;
      const humidity = payload[i]?.current?.relative_humidity_2m;
      if (!Number.isFinite(temp) || !Number.isFinite(humidity)) {
        throw new Error(`invalid weather reading for ${gu}`);
      }
      return { gu_nm: gu, temp: temp as number, humidity: humidity as number };
    });
  } catch {
    return null;
  }
}

export async function fetchOpenMeteoAir(
  gus: readonly string[],
): Promise<OpenMeteoAirRow[] | null> {
  try {
    const coords = coordinatesFor(gus);
    const payload = await fetchBatch<OpenMeteoAirResponse>(
      "https://air-quality-api.open-meteo.com/v1/air-quality",
      coords,
      "pm2_5",
    );
    return gus.map((gu, i) => {
      const pm25 = payload[i]?.current?.pm2_5;
      if (!Number.isFinite(pm25) || (pm25 as number) < 0) {
        throw new Error(`invalid PM2.5 reading for ${gu}`);
      }
      return { gu_nm: gu, pm25: pm25 as number };
    });
  } catch {
    return null;
  }
}

function coordinatesFor(gus: readonly string[]): GuCentroid[] {
  return gus.map((gu) => {
    const centroid = GU_CENTROIDS[gu];
    if (!centroid) throw new Error(`missing centroid for ${gu}`);
    return centroid;
  });
}

async function fetchBatch<T>(
  endpoint: string,
  coords: readonly GuCentroid[],
  current: string,
): Promise<T[]> {
  if (coords.length === 0) return [];

  const params = new URLSearchParams({
    latitude: coords.map(({ lat }) => lat).join(","),
    longitude: coords.map(({ lon }) => lon).join(","),
    current,
    timezone: "Asia/Seoul",
  });
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), 8_000);
  try {
    const response = await fetch(`${endpoint}?${params.toString()}`, {
      signal: ctl.signal,
    });
    if (!response.ok) throw new Error(`Open-Meteo HTTP ${response.status}`);
    const raw = (await response.json()) as T | T[];
    const payload = Array.isArray(raw) ? raw : [raw];
    if (payload.length !== coords.length) {
      throw new Error(`Open-Meteo returned ${payload.length}/${coords.length} rows`);
    }
    return payload;
  } finally {
    clearTimeout(timer);
  }
}
