import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { BotMessage } from "./Message";

const answer = {
  answer: "Standard dose: **1 g Q8H** [1]. HOD approval required [1].",
  sources: [
    { source_document: "drug_formulary.pdf", section_title: "1. Antimicrobials", collection: "clinical" },
    { source_document: "drug_formulary.pdf", section_title: "6. Renal Dose Adjustment", collection: "clinical" },
  ],
  retrieval_type: "hybrid_rag" as const,
  role: "doctor",
  access_denied: false,
};

describe("BotMessage", () => {
  it("labels the retrieval type and lists every source with document and section", () => {
    render(<BotMessage reply={answer} />);
    expect(screen.getByText("Hybrid RAG")).toBeInTheDocument();
    expect(screen.getAllByText("drug_formulary.pdf")).toHaveLength(2);
    expect(screen.getByText("1. Antimicrobials")).toBeInTheDocument();
    expect(screen.getByText("6. Renal Dose Adjustment")).toBeInTheDocument();
  });

  it("renders **bold** from the model as real emphasis", () => {
    render(<BotMessage reply={answer} />);
    expect(screen.getByText("1 g Q8H").tagName).toBe("STRONG");
  });

  it("labels SQL answers as SQL RAG and shows no sources block", () => {
    render(<BotMessage reply={{ ...answer, answer: "12 claims.", sources: [], retrieval_type: "sql_rag" }} />);
    expect(screen.getByText("SQL RAG")).toBeInTheDocument();
    expect(screen.queryByText(/Sources/)).not.toBeInTheDocument();
  });

  it("shows an access-restricted card for RBAC refusals", () => {
    render(
      <BotMessage
        reply={{
          ...answer,
          answer: "As a nurse, you don't have access to billing documents.",
          sources: [],
          access_denied: true,
        }}
      />,
    );
    expect(screen.getByText("Access restricted")).toBeInTheDocument();
    expect(screen.getByText(/As a nurse, you don't have access/)).toBeInTheDocument();
    expect(screen.queryByText("Hybrid RAG")).not.toBeInTheDocument();
  });
});

describe("answer formatting", () => {
  it("turns dash lines into a list and keeps citations inline", () => {
    render(<BotMessage reply={{ ...answer, answer: "Two things:\n- Dose **1 g** [1]\n- Approval needed [1]\n\nDone." }} />);
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
    expect(screen.getByText("Done.")).toBeInTheDocument();
    expect(screen.getAllByText("[1]")).toHaveLength(2);
  });
});

describe("ErrorMessage", () => {
  it("shows the backend's friendly message", async () => {
    const { ErrorMessage } = await import("./Message");
    render(<ErrorMessage text="Cannot reach the MediBot backend at http://localhost:8000. Is it running?" />);
    expect(screen.getByText(/Cannot reach the MediBot backend/)).toBeInTheDocument();
  });
});
