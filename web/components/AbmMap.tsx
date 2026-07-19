"use client";
/**
 * AbmMap — focused SEIR-V-D choropleth for the ABM dashboard (/map3d/abm).
 *
 * Colors each of the 25 gu by its infected fraction I(t)/N at the selected
 * simulation day, animated over the time slider. A gu highlighted by ARIA (or
 * clicked) gets a bright outline. Data is the precomputed trajectory bundle
 * abm-scenarios.json (export_abm_scenarios.py) — no live sim round trip.
 */
import DeckGL from "@deck.gl/react";
import { Map } from "react-map-gl/maplibre";
import "maplibre-gl/dist/maplibre-gl.css";
import { GeoJsonLayer } from "@deck.gl/layers";
import { useMemo } from "react";

export type Scenario = {
  label: string;
  legal_basis?: string;
  I_frac: number[][];
  city_incidence: number[];
  peak_day: number;
  attack_rate_pct: number[];
  city_attack_pct: number;
  deaths: number;
  epi_validity_ok: boolean;
};
export type AbmScenarios = {
  gu_names: string[];
  days: number;
  scenarios: Record<string, Scenario>;
};

const INITIAL_VIEW = {
  longitude: 126.99,
  latitude: 37.5665,
  zoom: 10.2,
  pitch: 0,
  bearing: 0,
};

/** Infected fraction → slate(low) → amber → red(high). */
function infColor(frac: number, max: number): [number, number, number, number] {
  const t = max > 0 ? Math.min(1, frac / max) : 0;
  const r = Math.round(50 + t * 200);
  const g = Math.round(70 + Math.sin(Math.min(t, 0.6) / 0.6 * Math.PI) * 120);
  const b = Math.round(110 * (1 - t));
  return [r, g, b, Math.round(55 + t * 185)];
}

export function AbmMap({
  data,
  geojson,
  scenario,
  day,
  highlightedGu,
  onGuClick,
}: {
  data: AbmScenarios | null;
  geojson: GeoJSON.FeatureCollection | null;
  scenario: string;
  day: number;
  highlightedGu: string | null;
  onGuClick?: (gu: string) => void;
}) {
  const sc = data?.scenarios[scenario] ?? null;
  const guIndex = useMemo(
    () => Object.fromEntries((data?.gu_names ?? []).map((n, i) => [n, i])),
    [data],
  );
  const maxFrac = useMemo(
    () => (sc ? Math.max(1e-9, ...sc.I_frac.map((row) => Math.max(...row))) : 1),
    [sc],
  );

  const layers = [
    geojson && sc
      ? new GeoJsonLayer({
          id: "abm-seir",
          data: geojson,
          stroked: true,
          filled: true,
          getFillColor: (f: GeoJSON.Feature) => {
            const i = guIndex[(f.properties?.name as string) ?? ""];
            return i != null
              ? infColor(sc.I_frac[day]?.[i] ?? 0, maxFrac)
              : ([0, 0, 0, 0] as [number, number, number, number]);
          },
          getLineColor: (f: GeoJSON.Feature) =>
            (f.properties?.name === highlightedGu
              ? [255, 230, 80, 255]
              : [120, 140, 180, 120]) as [number, number, number, number],
          getLineWidth: (f: GeoJSON.Feature) =>
            f.properties?.name === highlightedGu ? 90 : 18,
          lineWidthUnits: "meters",
          lineWidthMinPixels: 1,
          lineWidthMaxPixels: 5,
          pickable: true,
          onClick: (info: { object?: GeoJSON.Feature }) =>
            info.object && onGuClick?.(info.object.properties?.name as string),
          updateTriggers: {
            getFillColor: [day, scenario, maxFrac],
            getLineColor: [highlightedGu],
            getLineWidth: [highlightedGu],
          },
        })
      : null,
  ].filter(Boolean);

  return (
    <DeckGL initialViewState={INITIAL_VIEW} controller={true} layers={layers}>
      <Map
        mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
        reuseMaps
      />
    </DeckGL>
  );
}
