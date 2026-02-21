////////// Chapter 2 - Operators //////////
/*
 * JavaScript supports many operators for performing operations:
 * - Arithmetic: For mathematical calculations
 * - Assignment: For assigning values to variables
 * - Comparison: For comparing values (not shown)
 * - Logical: For boolean logic
 * - Bitwise: For bit-level operations
 */

// 1. Arithmetic Operators
let num1 = 10;
let num2 = 5;
let sum = num1 + num2;       // Addition
let diff = num1 - num2;      // Subtraction
let prod = num1 * num2;      // Multiplication
let div = num1 / num2;       // Division
let mod = num1 % num2;       // Modulus (remainder)
let exp = num1 ** num2;      // Exponentiation (10^5)
console.log("Sum: " + sum);
console.log("Difference: " + diff);
console.log("Product: " + prod);
console.log("Division: " + div);
console.log("Modulus: " + mod);
console.log("Exponentiation: " + exp);

// 2. Assignment Operators
let a = 10;
let b = 5;
a += b;                      // Add and assign
console.log("a += b: " + a); // 15
a -= b;                      // Subtract and assign
console.log("a -= b: " + a); // 10
a *= b;                      // Multiply and assign
console.log("a *= b: " + a); // 50
a /= b;                      // Divide and assign
console.log("a /= b: " + a); // 10
a %= b;                      // Modulus and assign
console.log("a %= b: " + a); // 0
a **= b;                     // Exponentiate and assign
console.log("a **= b: " + a); // 0^5 = 0

// 3. Logical Operators
let c = true;
let d = false;
let and = c && d;            // logical AND - true if both operands are true
let or = c || d;             // logical OR - true if at least one operand is true
let not = !c;                // logical NOT - inverts the boolean value
console.log("Logical AND: " + and); // false
console.log("Logical OR: " + or);   // true
console.log("Logical NOT: " + not); // false

// 4. Bitwise Operators
console.log("a && b: " + (a && b)); // 0 (logical AND with non-boolean values)
console.log("a || b: " + (a || b)); // 5 (logical OR with non-boolean values)
console.log("a^b: " + (a ^ b));     // 5 (bitwise XOR)
console.log("shift left: " + (22 << 2)); // 88 (22 * 2^2)
console.log("shift right: " + (22 >> 2)); // 5 (22 / 2^2)