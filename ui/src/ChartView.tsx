// Chart rendering for assistant replies — the same component the embeddable
// widget uses (frontend/src/AssistantWidgetApp.tsx), copied so ui/ stands alone.
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Chart } from "./api";

const CHART_COLORS = [
  "#4f46e5",
  "#22c55e",
  "#f59e0b",
  "#ef4444",
  "#06b6d4",
  "#a855f7",
  "#ec4899",
  "#84cc16",
];

function buildChartRows(chart: Chart): Record<string, string | number>[] {
  return chart.labels.map((label, index) => {
    const row: Record<string, string | number> = { name: label };
    for (const series of chart.series) {
      row[series.name] = series.values[index] ?? 0;
    }
    return row;
  });
}

export function ChartView({ chart }: { chart: Chart }) {
  const data = buildChartRows(chart);
  const showLegend = chart.series.length > 1;

  return (
    <div className="aw-chart">
      <div className="aw-chart-title">{chart.title}</div>
      <ResponsiveContainer width="100%" height={220}>
        {chart.chartType === "bar" ? (
          <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            {showLegend && <Legend wrapperStyle={{ fontSize: 11 }} />}
            {chart.series.map((series, index) => (
              <Bar
                key={series.name}
                dataKey={series.name}
                fill={CHART_COLORS[index % CHART_COLORS.length]}
              />
            ))}
          </BarChart>
        ) : chart.chartType === "line" ? (
          <LineChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            {showLegend && <Legend wrapperStyle={{ fontSize: 11 }} />}
            {chart.series.map((series, index) => (
              <Line
                key={series.name}
                type="monotone"
                dataKey={series.name}
                stroke={CHART_COLORS[index % CHART_COLORS.length]}
              />
            ))}
          </LineChart>
        ) : (
          <PieChart margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Pie
              data={data}
              dataKey={chart.series[0]?.name ?? "value"}
              nameKey="name"
              cx="50%"
              cy="50%"
              outerRadius={80}
              label
            >
              {data.map((_, index) => (
                <Cell key={index} fill={CHART_COLORS[index % CHART_COLORS.length]} />
              ))}
            </Pie>
          </PieChart>
        )}
      </ResponsiveContainer>
      {chart.sourceChunks.length > 0 && (
        <div className="aw-citations aw-chart-citations">
          {/* Document-level only, same display name as the reply's source chips. */}
          {[...new Set(chart.sourceChunks.map((c) => c.documentTitle))].map((title) => (
            <div key={title}>Chart source: {title.replace(/_/g, " ")}</div>
          ))}
        </div>
      )}
    </div>
  );
}
