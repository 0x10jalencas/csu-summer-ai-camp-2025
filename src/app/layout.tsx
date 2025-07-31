import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Student Success Portal",
  description: "Monitor student progress, identify risks, and drive positive outcomes",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
