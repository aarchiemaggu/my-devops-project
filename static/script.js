const organs = document.querySelectorAll(".organ");
const dropzones = document.querySelectorAll(".dropzone");

organs.forEach(organ => {
    organ.addEventListener("dragstart", e => {
        e.dataTransfer.setData("text", organ.id);
    });
});

dropzones.forEach(zone => {
    zone.addEventListener("dragover", e => {
        e.preventDefault();
    });

    zone.addEventListener("drop", e => {
        const organId = e.dataTransfer.getData("text");
        const organ = document.getElementById(organId);
        if (organ) {
            e.target.innerHTML = "";
            e.target.appendChild(organ);
        }
    });
});
