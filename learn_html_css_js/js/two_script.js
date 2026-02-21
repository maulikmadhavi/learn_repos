const form = document.getElementById('user-form');
const result = document.getElementById('result');

form.addEventListener('submit', (e) => {
  e.preventDefault();

  const name = document.getElementById('name').value;
  const age = document.getElementById('age').value;
  const place = document.getElementById('place').value;
  const gender = document.getElementById('gender').value;

  result.innerHTML = `
    <h2>User Details:</h2>
    <p>Name: ${name}</p>
    <p>***Age: ${age} years old</p>
    <p>***Place: ${place}</p>
    <p>***Gender: ${gender}</p>
  `;
});


form.addEventListener('mouseover', (event) => {
  console.log(`You hovered over the button!`);
});