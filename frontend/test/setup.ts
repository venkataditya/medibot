import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// vitest does not expose afterEach globally, so Testing Library cannot auto-clean
afterEach(cleanup);
