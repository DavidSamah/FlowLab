import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FlowLab",
  description: "An explainable 2D fluid-flow research prototype",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
