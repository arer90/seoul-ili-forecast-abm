/**
 * Ambient declarations for the `leaflet-contextmenu` plugin.
 *
 * The package ships no types; we dynamically `import("leaflet-contextmenu")`
 * for its side-effect of extending `L.Map` options at runtime. TS only needs
 * to know the module resolves. Per-option widening is already handled by
 * `@ts-expect-error` annotations at the call sites in MapPanel.tsx.
 */
declare module "leaflet-contextmenu";
