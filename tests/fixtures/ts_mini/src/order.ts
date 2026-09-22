import { formatAmount, parseAmount } from "./util";

export interface OrderLine {
  sku: string;
  cents: number;
}

export class Cart {
  private lines: OrderLine[] = [];

  add(line: OrderLine): void {
    this.lines.push(line);
  }

  totalCents(): number {
    return this.lines.reduce((acc, l) => acc + l.cents, 0);
  }
}

export function renderCart(cart: Cart): string {
  return \`total=\${formatAmount(cart.totalCents())}\`;
}

export function buildCart(raw: string): Cart {
  const cart = new Cart();
  const cents = parseAmount(raw);
  cart.add({ sku: "X", cents });
  return cart;
}
