import { Cart, renderCart } from "./order";

const cart = new Cart();
cart.add({ sku: "BOOT", cents: 19900 });

console.log(renderCart(cart));
