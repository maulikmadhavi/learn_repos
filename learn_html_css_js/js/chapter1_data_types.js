////////// Chapter 1 - Data Types //////////
/*
 * JavaScript has several primitive data types:
 * - Number: For numeric values (integers and floating-point)
 * - String: For text values
 * - Boolean: For true/false values
 * - Undefined: Variables declared but not assigned
 * - Null: Represents the intentional absence of any value
 * - Symbol and BigInt (not shown in examples)
 */

var i;
i = 2;
console.log(i);

// Variable declaration differences:
// var is function scoped, let is block scoped
// var can be redeclared, let cannot

// 1. Number
let x = 5;
console.log(typeof x); // number

// 2. String
let y = "Hello";
console.log(typeof y); // string

// 3. Boolean
let z = true;
console.log(typeof z); // boolean

// 4. Undefined - variable declared but not assigned a value
let a; // undefined
console.log(typeof a); // undefined

// 5. Null - intentional absence of any object value
let b = null; // null
console.log(typeof b); // object (this is a known JavaScript quirk)