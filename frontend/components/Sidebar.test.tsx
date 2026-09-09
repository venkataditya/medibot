import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { Sidebar } from "./Sidebar";

const session = {
  token: "t",
  username: "nurse.priya",
  role: "nurse",
  collections: ["general", "nursing"],
  sql_access: false,
};

describe("Sidebar", () => {
  it("shows the role badge, the user, and marks inaccessible collections", () => {
    render(<Sidebar session={session} onSignOut={() => {}} />);
    expect(screen.getByText("Nurse")).toBeInTheDocument();
    expect(screen.getByText(/nurse\.priya/)).toBeInTheDocument();
    expect(screen.getByText("general")).not.toHaveClass("off");
    expect(screen.getByText("billing")).toHaveClass("off");
    expect(screen.getByText("equipment")).toHaveClass("off");
    expect(screen.getByText(/not available for this role/)).toBeInTheDocument();
  });

  it("tells analytical roles that claims and ticket analytics are available", () => {
    render(
      <Sidebar
        session={{ ...session, role: "billing_executive", collections: ["general", "billing"], sql_access: true }}
        onSignOut={() => {}}
      />,
    );
    expect(screen.getByText("Billing executive")).toBeInTheDocument();
    expect(screen.getByText(/available/)).toBeInTheDocument();
  });

  it("signs out on click", () => {
    const onSignOut = vi.fn();
    render(<Sidebar session={session} onSignOut={onSignOut} />);
    screen.getByRole("button", { name: /sign out/i }).click();
    expect(onSignOut).toHaveBeenCalled();
  });
});
