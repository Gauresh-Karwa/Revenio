export const api = async <T,>(path: string, init?: RequestInit): Promise<T> => {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 12_000);
  try {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...init,
      signal: init?.signal ?? controller.signal,
    });
    if (!response.ok) throw new Error(await response.text());
    return response.json() as Promise<T>;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("The Revenio API did not respond within 12 seconds.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
};

export const money = (amount: number) => new Intl.NumberFormat("en-IN", {
  style: "currency", currency: "INR", maximumFractionDigits: 0,
}).format(amount);
