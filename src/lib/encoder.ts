import schema from "@/lib/feature_schema.json";

type RawRow = Record<string, any>;
type FeatureSpec = (typeof schema)["features"][number];

function encodeValue(spec: FeatureSpec, raw: any): number {
  if (spec.type === "numeric") {
    const n = Number(raw);
    return isNaN(n) ? 0 : n;
  }
  const map = spec.mapping as unknown as Record<string, number>;
  return map[raw] ?? map["Did not answer"] ?? 0;
}

export function encodeRow(row: RawRow): number[] {
  return schema.order.map((name) => {
    const spec = schema.features.find((f) => f.name === name)!;
    return encodeValue(spec, row[name]);
  });
}